cd src
torchrun --nproc_per_node 1 \
    --master_addr=127.0.0.3 --master_port=29117 \
    -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data "/home/corsair/DATA_SCRIPTS/Thesis/Datasets/cc12m.csv" \
    --val-data "/home/corsair/DATA_SCRIPTS/Thesis/Datasets/cc3_val.csv" \
    --data-root "/home/corsair/DATA_SCRIPTS/Thesis/Datasets/dataset/" \
    --val-data-root "/home/corsair/DATA_SCRIPTS/Thesis/Datasets/kd/" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val="/home/corsair/DATA_SCRIPTS/Thesis/Datasets/imagenet-val/" \
    --resume "/media/corsair/Expansion/Thesis/Scripts/CLIP-KD/Weights_Results/VimCLIP_Models/baseline_vim_bidirectional_b_cc12m/2025_11_12-22_46_34-model_ViT-B-16-lr_0.001-b_64-epochs_64-tag_baseline-mamba/checkpoints/epoch_64.pt" \
    --warmup 1000 \
    --eval \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 64 \
    --workers=24 \
    --model ViT-B-16 \
    --tag baseline-mamba-vimt-attention
    #--logs "./logs-evaluation-results-kd-fd/"  \
    #--pretrained "/home/corsair/DATA_SCRIPTS/Thesis/CLIP-KD/vim_t_midclstok_76p1acc.pth" \
