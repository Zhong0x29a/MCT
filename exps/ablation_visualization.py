
"""
CF100-ipc1
expert_step	40	30
8	24.49	23.65
6	23.78	23.34
5	23.39	23.3

CF100-ipc10
	30
8	40.95
5	40.89
4	41.38
3	41.64
2	41.4

CF100-ipc50
	50
8	43.74
6	44.6
4	46.29
3	46.62
2	46.02


"""
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = {
        "CF100-ipc1":{
            "40":{
                "8":24.49,
                "6":23.78,
                "5":23.39
            },
            # "30":{
            #     "8":23.65,
            #     "6":23.34,
            #     "5":23.3
            # }
        },
        "CF100-ipc10":{
            "30":{
                "8":40.95,
                "5":40.89,
                "4":41.38,
                "3":41.64,
                "2":41.4
            },
        },
        "CF100-ipc50":{
            "50":{
                "8":43.74,
                "6":44.6,
                "4":46.29,
                "3":46.62,
                "2":46.02
            },
        }
        
    }
    
    # plot 折线图
    # Plotting
    for label, outer_dict in data.items():
        for inner_dict in outer_dict.values():
            x = list(map(int, inner_dict.keys()))
            y = list(inner_dict.values())
            plt.plot(x, y, marker='o', label=label)
    
    plt.xlim(2, 8)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Line Chart')
    plt.legend()
    plt.show()

