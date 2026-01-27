import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap.umap_ as umap
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, KDClipLoss, get_cast_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

    
def train_kd_one_epoch(model, t_model, data, epoch, loss, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_task = AverageMeter()
    loss_icl = AverageMeter()
    loss_ckd = AverageMeter()
    loss_cross_kd  = AverageMeter()
    loss_fd = AverageMeter()
    loss_gd = AverageMeter()
    loss_afd = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts, distill=True, mask_ratio=args.mask_ratio)

            with torch.no_grad():
                t_image_features, t_text_features, t_logit_scale = t_model(images, texts)

            losses = loss(image_features, text_features, logit_scale, \
                t_image_features, t_text_features, t_logit_scale)
             
            task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss = losses
            total_loss = task_loss + ckd_loss + icl_loss + cross_kd_loss + fd_loss + gd_loss + afd_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_task.update(task_loss.item(), batch_size)
            loss_icl.update(icl_loss.item(), batch_size)
            loss_ckd.update(ckd_loss.item(), batch_size)
            loss_cross_kd.update(cross_kd_loss.item(), batch_size)
            loss_fd.update(fd_loss.item(), batch_size)
            loss_gd.update(gd_loss.item(), batch_size)
            loss_afd.update(afd_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Task Loss: {loss_task.val:#.5g} ({loss_task.avg:#.4g}) "
                f"ICL Loss: {loss_icl.val:#.5g} ({loss_icl.avg:#.4g}) "
                f"CKD Loss: {loss_ckd.val:#.5g} ({loss_ckd.avg:#.4g}) "
                f"Cross KD Loss: {loss_cross_kd.val:#.5g} ({loss_cross_kd.avg:#.4g}) "
                f"FD Loss: {loss_fd.val:#.5g} ({loss_fd.avg:#.4g}) "
                f"GD Loss: {loss_gd.val:#.5g} ({loss_gd.avg:#.4g}) "
                f"AFD Loss: {loss_afd.val:#.5g} ({loss_afd.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    
    
def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    print(zero_shot_metrics)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    print("starting with the inference out here")
    
    dataloader = data['val'].dataloader
    single_batch = next(iter(dataloader))
    images, texts = single_batch

    images = images.to(device=device, dtype=cast_dtype)
    texts = texts.to(device=device)       
    
    start_time = time.time()
    with torch.no_grad():
        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            # Extract attention map using your separate Vision Mamba function per image, here taking the 4th image (index 3)
            #print(model.visual)
            #attn_map, _ = generate_raw_attn(model.visual, images[3].unsqueeze(0), start_layer=15)
    end_time = time.time()

    print(f"Inference time: {end_time - start_time:.3f}s")
    #print(f"Attention map shape: {attn_map.shape}")

    # attn_map shape assumed: [1, num_patches], flatten excluding CLS token
    #cls_attn_map = attn_map.squeeze().cpu()  # shape: [num_patches]

    # Compute patch grid size assuming square grid (e.g., 14x14 for 224x224 patched by 16)
    #patch_grid_size = int(cls_attn_map.shape[0] ** 0.5)
    #cls_attn_map_2d = cls_attn_map.reshape(patch_grid_size, patch_grid_size)

    # Upsample attention map to image spatial dimensions (H, W)
    #mg = images[3].cpu()  # Image tensor [3, H, W]
    #attn_resized = TF.resize(cls_attn_map_2d.unsqueeze(0).unsqueeze(0), size=mg.shape[1:], interpolation=TF.InterpolationMode.BILINEAR)
    #attn_resized = attn_resized.squeeze().numpy()

    # Normalize attention to [0,1]
    #attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    # Normalize image to [0,1]
    #img_np = mg.permute(1, 2, 0).numpy()
    #img_min, img_max = img_np.min(), img_np.max()
    #img_np = (img_np - img_min) / (img_max - img_min + 1e-8)

    # Plot original image
    #plt.imshow(img_np)
    # Overlay attention heatmap
    #plt.imshow(attn_resized, cmap='jet', alpha=0.5)
    #plt.axis('off')
    #plt.tight_layout()

    # Save overlay visualization
    #save_path = os.path.join(args.checkpoint_path, f"attention_epoch{epoch}.png")
    #plt.savefig(save_path, bbox_inches='tight')
    #plt.close()
    
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")
            # Concatenate all embeddings
            image_embeds = torch.cat(all_image_features).numpy()
            text_embeds = torch.cat(all_text_features).numpy()
            #print(image_embeds.shape, text_embeds.shape)
            # Combine image and text embeddings
            combined_embeds = np.concatenate([image_embeds, text_embeds], axis=0)
            num_images = image_embeds.shape[0]
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            embeds_2d = reducer.fit_transform(combined_embeds)
            reduction_name = 'UMAP'

            # Separate back to image/text
            image_embeds_2d = embeds_2d[:num_images]
            text_embeds_2d = embeds_2d[num_images:]

            # Centroid and distance in 2D UMAP space
            image_centroid_2d = image_embeds_2d.mean(axis=0)
            text_centroid_2d = text_embeds_2d.mean(axis=0)
            centroid_dist_2d = np.linalg.norm(image_centroid_2d - text_centroid_2d)
            #print(f"2D UMAP space: Euclidean {centroid_dist_2d:.4f}")
            
            # Scatter plot: images '+', text 'x'
            plt.figure(figsize=(10, 10))
            plt.scatter(image_embeds_2d[:, 0], image_embeds_2d[:, 1], label='Images', alpha=0.6, c='blue', marker='+')
            plt.scatter(text_embeds_2d[:, 0], text_embeds_2d[:, 1], label='Text', alpha=0.4, c='red', marker='x')
            plt.legend(loc='best')
            plt.title(f'Embedding Space Visualization ({reduction_name})')
            plt.xlabel('Dimension-1')
            plt.ylabel('Dimension-2')
            plt.savefig(os.path.join(args.checkpoint_path, f'embeddings.png'))
            plt.close()

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


