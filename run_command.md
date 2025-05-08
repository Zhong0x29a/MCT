```bush
export CUDA_VISIBLE_DEVICES=0
```

# Original MTT

## CIFAR-10

### no zca

1. ipc=1, no zca
```bush
nohup python -u distill.py --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=5 --lr_img=1e3 --lr_lr=1e-07 --lr_init=1e-2 > distill_cifar10_ipc1.log 2>&1 &
```

1. ipc=10, no zca
```bush
nohup python -u distill.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=15 --lr_img=1e3 --lr_lr=1e-05 --lr_init 1e-2 > distill_cifar10_ipc10.log 2>&1 &
```

1. ipc=50, no zca
```bush
nohup python -u distill.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --lr_img=1e3 --lr_lr=1e-05 --lr_init 1e-3 > distill_cifar10_ipc50.log 2>&1 &
```

### with zca

1. ipc=1, zca
```bush
nohup python distill.py --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=2 --lr_img=1e2 --lr_lr=1e-07 --lr_init 1e-2 --zca > distill_cifar10_ipc1_zca.log 2>&1 &
```

1. ipc=10, zca
```bush
nohup python distill.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --lr_img=1e5 --lr_lr=1e-04 --lr_init 1e-2 --zca > distill_cifar10_ipc10_zca.log 2>&1 &
```

1. ipc=50, zca
```bush
nohup python distill.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --lr_img=1e2 --lr_lr=1e-05 --lr_init 1e-3 --zca > distill_cifar10_ipc50_zca.log 2>&1 &
```

## CIFAR-100 (default use zca)

generate buffer
```bush
nohup python buffer.py --dataset=CIFAR100 --zca 
```

1. ipc=1
```bush
nohup python distill.py --dataset=CIFAR100 --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --lr_img=1e3 --lr_lr=1e-05 --lr_init 1e-2 --zca > distill_cifar100_ipc1_zca.log 2>&1 &
```

### no zca
...

## TinyImageNet

generate buffer
```bush
nohup python buffer.py --dataset=Tiny --data_path=./data/tiny-imagenet-200 --zca
```


## Convexed MTT

## CIFAR-10

### no zca

1. ipc=1, no zca
```bush
nohup python -u distill_convexified.py --dataset=CIFAR10 --ipc=1 --syn_steps=60 --expert_epochs=6 --max_start_epoch=4 --lr_img=1e3 --lr_lr=1e-07 --lr_init=1e-2 > distill_cifar10_ipc1.log 2>&1 &
```

1. ipc=10, no zca
```bush
nohup python -u distill_convexified.py --convex_ver=1 --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=15 --max_start_epoch=8 --lr_img=1e3 --lr_lr=1e-05 --lr_init 1e-2 > distill_cifar10_ipc10.log 2>&1 &
```

1. ipc=50, no zca
```bush
nohup python -u distill_convexified.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --lr_img=1e3 --lr_lr=1e-05 --lr_init 1e-3 > distill_cifar10_ipc50.log 2>&1 &
```
