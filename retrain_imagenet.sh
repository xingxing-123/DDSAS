CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train_imagenet.py \
    --tmp_data_dir data \
    --auxiliary \
    --arch DDSAS_imagenet \
    --batch_size 1024 \
    --init_channels 46 \
    --learning_rate 0.5
