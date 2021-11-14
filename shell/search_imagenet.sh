CUDA_VISIBLE_DEVICES=7 \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  \
    train_search_imagenet.py \
    --space_config space_config_imagenet \
    --data data/imagenet_split/ \
    --batch_size 512 \
    --learning_rate=0.5 \
    --learning_rate_min=0 \
    --arch_learning_rate=6e-3 \
    --arch_weight_decay=1e-3 \
    --begin=0 \
    --epochs=60
