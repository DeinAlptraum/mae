IMAGENET_DIR=/mnt/data0/jannick/img1k
GPU="cuda:3"
CHECKPOINT=jobdir/preencoder/checkpoint-4.pth

python main_finetune.py --mask_method preencoder --finetune $CHECKPOINT \
    --model vit_large_patch16 --data_path ${IMAGENET_DIR} --device $GPU \
    --batch_size 32 --epochs 100 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval