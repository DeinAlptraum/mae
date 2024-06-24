IMAGENET_DIR=/mnt/data0/jannick/img1k
GPU="cuda:3"

python main_finetune.py --eval --resume jobdir/patches/checkpoint-4.pth \
    --model vit_large_patch16 --batch_size 16 \
    --data_path ${IMAGENET_DIR} --device "cuda:$1" \
    --mask_method patches
