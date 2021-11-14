python train_search.py \
    --data data/cifar100 \
    --dataset cifar100 \
    --space_config space_config_shrink \
    --arch_learning_rate 3e-5 \
    --arch_weight_decay 0 \
    --arch DDSAS_cifar100_search \
    --saliency_type simple \
    --dss_max_ops 28 \
    --dss_freq 30 \
    --epochs=60 \
    --seed 0 \
    --gpu 5

