```bush
export CUDA_VISIBLE_DEVICES=0
```

## Generate buffer
```bush
nohup python buffer.py --dataset=CIFAR10 --zca 

nohup python buffer.py --dataset=CIFAR100 --zca 

nohup python buffer.py --dataset=Tiny --data_path=./data/tiny-imagenet-200 --zca
```


## Generate Convexified Trajectory

```bush
nohup python -u trajectory_compression.py --num_interpolation=4  --zca  --dataset=CIFAR10
```

```bush
nohup python -u trajectory_compression.py --num_interpolation=4  --zca  --dataset=CIFAR100
```

```bush
nohup python -u trajectory_compression.py  --num_interpolation=4  --zca  --dataset=Tiny --data_path=../data/tiny-imagenet-200
```

## Distill

### CIFAR-10

no zca

1. ipc=1, no zca
```bush
nohup python -u distill_convexified.py --dataset=CIFAR10 --ipc=1 --syn_steps=60 --expert_epochs=6 --max_start_epoch=4 --lr_img=1e3 --lr_lr=1e-07 --lr_init=1e-2
```

1. ipc=10, no zca
```bush
nohup python -u distill_convexified.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=15 --max_start_epoch=8 --lr_img=1e3 --lr_lr=1e-05
```

1. ipc=50, no zca
```bush
nohup python -u distill_convexified.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --lr_img=1e3 --lr_lr=1e-05 --lr_init 1e-3
```
