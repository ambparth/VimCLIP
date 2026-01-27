import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.manifold import TSNE
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template, imagenet_a, imagenet_r_indices

def unique_colors(num_colors):
    """Generate num_colors visually distinct colors using HSV space."""
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    colors_hsv = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
    colors_rgb = hsv_to_rgb(colors_hsv)
    return colors_rgb

def save_legend(classnames, colors, legend_save_path="tsne_legend.png", cols=5):
    num_classes = len(classnames)
    rows = math.ceil(num_classes / cols)
    fig, ax = plt.subplots(figsize=(cols * 3, rows * 0.3))
    ax.axis('off')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=7)
               for i in range(num_classes)]

    legend = ax.legend(handles, classnames, ncol=cols, frameon=False, loc='center',
                       fontsize=6, handletextpad=0.5, columnspacing=0.5)
    fig.savefig(legend_save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            if args.distributed and not args.horovod:
                if hasattr(model, 'module'):
                    class_embeddings = model.module.encode_text(texts)
                else:
                    class_embeddings = model.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classifier, dataloader, args, name,
        tsne_save_path="tsne_plot.png",
        legend_save_path="tsne_legend.png"):

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    
    if name == 'imagenet-a':
        imagenet_a_indices = [k for k in imagenet_a if imagenet_a[k] != -1]

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)
            with autocast():
                if args.distributed and not args.horovod:
                    if hasattr(model, 'module'):
                        image_features = model.module.encode_image(images)
                    else:
                        image_features = model.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            
            all_embeddings.append(image_features.cpu())
            all_labels.append(target.cpu())

            if name == 'imagenet-r':
                logits = logits[:, imagenet_r_indices]
            if name == 'imagenet-a':
                logits = logits[:, imagenet_a_indices]

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    classes_to_plot = list(range(10))
    indices = np.isin(all_labels, classes_to_plot)
    filtered_embeddings = all_embeddings[indices]
    filtered_labels = all_labels[indices]

    # Apply TSNE on filtered embeddings
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(filtered_embeddings)

    # Define colors for 10 classes
    num_classes = 10
    colors = unique_colors(num_classes)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(filtered_labels)
    for cls in unique_labels:
        cls_indices = filtered_labels == cls
        plt.scatter(embeddings_2d[cls_indices, 0], embeddings_2d[cls_indices, 1],
                    alpha=0.6, color=colors[cls], s=15, label=imagenet_classnames[cls])

    plt.title(f"t-SNE Visualization of First 10 Classes on {name}")
    plt.legend(loc='best', fontsize=10)  # Legend inside plot
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    tsne_save_path = "tsne_10_classes_plot.png"
    plt.savefig(tsne_save_path)

    legend_save_path = "tsne_10_classes_legend.png"
    save_legend(imagenet_classnames[:10], colors, legend_save_path)

    return top1, top5

def zero_shot_eval(model, data, epoch, args):
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')
    classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args, 'imagenet-val')
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args, 'imagenet-v2')
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'imagenet-r' in data:
        top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args, 'imagenet-r')
        results['imagenet-r-zeroshot-val-top1'] = top1
        results['imagenet-r-zeroshot-val-top5'] = top5
    if 'imagenet-a' in data:
        top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args, 'imagenet-a')
        results['imagenet-a-zeroshot-val-top1'] = top1
        results['imagenet-a-zeroshot-val-top5'] = top5
    if 'imagenet-sketch' in data:
        top1, top5 = run(model, classifier, data['imagenet-sketch'].dataloader, args, 'imagenet-sketch')
        results['imagenet-sketch-zeroshot-val-top1'] = top1
        results['imagenet-sketch-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
