python train_search.py \
    --data data/cifar10 \
    --dataset cifar10 \
    --space_config space_config_shrink \
    --arch DDSAS_cifar10_search \
    --saliency_type simple \
    --dss_max_ops 28 \
    --dss_freq 30 \
    --epochs=60 \
    --seed 0 \
    --gpu 0

