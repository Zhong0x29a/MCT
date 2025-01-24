

"""
CIFAR-10:
ours: 532881660 bytes
mtt: 6547733490 bytes
CIFAR-100:
ours: 827948540 bytes
mtt: 10309830130 bytes
Tiny:
ours: 3121862140 bytes
mtt: 39557228530 bytes


"""
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = {
        "CIFAR10": {
            "mtt": 6547733490,
            "ours": 532881660,
        },
        "CIFAR100": {
            "mtt": 10309830130,
            "ours": 827948540,
        },
        "Tiny": {
            "mtt": 39557228530,
            "ours": 3121862140,
        }
    }
    
    '''
    MTT:
    Cifar10:    6547733490.0
    Cifar100:   10309830130.0
    Tiny:       39557228530.0
    
    DATM:
    Cifar10:    2619093396.0
    Cifar100:   4123932052.0
    Tiny:       15822891412.0
    
    MCT:
    Cifar10:    532881660.0
    Cifar100:   827948540.0
    Tiny:       3121862140.0
    
    
    ----MB----
    
    MTT:
    Cifar10:    6244.41
    Cifar100:   9832.22
    Tiny:       37724.71
    
    DATM:
    Cifar10:    2497.76
    Cifar100:   3932.89
    Tiny:       15089.89
    
    MCT:
    Cifar10:    508.2
    Cifar100:   789.59
    Tiny:       2977.24
    
    '''
    
    print(data["CIFAR10"]["ours"] / data["CIFAR10"]["mtt"])
    print(data["CIFAR100"]["ours"] / data["CIFAR100"]["mtt"])
    print(data["Tiny"]["ours"] / data["Tiny"]["mtt"])
    
    print(data["CIFAR10"]["mtt"] / data["CIFAR10"]["ours"])
    print(data["CIFAR100"]["mtt"] / data["CIFAR100"]["ours"])
    print(data["Tiny"]["mtt"] / data["Tiny"]["ours"])
    
    # plot 柱状图，ours为蓝色#59c4da，mtt为橙色
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    x = np.arange(len(data))
    width = 0.35
    for i, (k, v) in enumerate(data.items()):
        ax.bar(x[i] - width/2, v["mtt"], width, label=f"{k} mtt", color='orange')
        ax.bar(x[i] + width/2, v["ours"], width, label=f"{k} ours", color='#40c0ca')
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys())
    ax.set_ylabel('Bytes')  # 设置y轴标签为'Bytes'
    ax.legend()
    plt.show()
