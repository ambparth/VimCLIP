cd src

torchrun --nproc_per_node=1 \
    --master_addr=127.0.0.2 --master_port=29514 \
    -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data "/data/ambarish/CLIP-KD/cc3_train.csv,/data/ambarish/CLIP-KD/cc12m.csv" \
    --val-data "/data/ambarish/CLIP-KD/cc3_val.csv" \
    --data-root "/data/ambarish/kd/,/data/ambarish/CLIP-KD/datasets/" \
    --val-data-root "/data/ambarish/kd/" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val="/data/ambarish/CLIP-KD/imagenet-val" \
    --warmup 3000 \
    --batch-size 128 \
    --lr 1e-3 \
    --logs "/data/ambarish/logs-mvision/" \
    --wd 0.1 \
    --epochs 32 \
    --workers 48 \
    --model ViT-B-16\
    --precision "fp32" \
    --tag baseline-mamba
    #--pretrained "/home/ambarish/KDistill/CLIP-KD/vim_b_midclstok_81p9acc.pth" \
#--imagenet-val "/data/ambarish/CLIP-KD/imagenet-val/" \
    #--pretrained "/data/ambarish/GlobalScripts/CLIP-KD/vim_s_midclstok_80p5acc.pth" \
    #--imagenet-a="/home/corsair/DATA_SCRIPTS/Thesis/Datasets/imagenet-a" \
    #--imagenet-r="/home/corsair/DATA_SCRIPTS/Thesis/Datasets/imagenet-r" \
    #--imagenet-sketch="/home/corsair/DATA_SCRIPTS/Thesis/Datasets/imagenet-sketch/sketch/" \
