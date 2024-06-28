#JOB_DIR=../jobdir
IMAGENET_DIR=/mnt/data0/jannick/img1k
GPU="cuda:3"

python main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --device $GPU \
    --data_path ${IMAGENET_DIR} \
    --coverage_ratio 15 \
    --mask_method preencoder
    #--resume jobdir/preencoder/checkpoint-4.pth


# python submitit_pretrain.py \
#     --nodes 1 \
#     --ngpus 1 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --local \
#     --mask_ratio 0.75 \
#     --epochs 20 \
#     --warmup_epochs 1 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path ${IMAGENET_DIR}
