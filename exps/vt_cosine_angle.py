import torch

if __name__ == '__main__':
    # sftp://root@211.87.232.86:20025/root/zhongwenliang/DC-MTT/buffers/CIFAR10/ConvNet/replay_buffer_0.pt
    buffer_path = '/root/zhongwenliang/DC-MTT/buffers/CIFAR10/ConvNet/replay_buffer_0.pt'
    
    traj_buffer = torch.load(buffer_path)[0]
    
    # print(traj_buffer)
    
    V_Ts = []
    
    STEP_SIZE = 1
    for i in range(STEP_SIZE, len(traj_buffer)):
        V_T = []
        for j in range(len(traj_buffer[i])):
            V_T.append(traj_buffer[i][j] - traj_buffer[i-STEP_SIZE][j])
        V_T = torch.cat([layer.flatten() for layer in V_T], dim=0).flatten()
        V_Ts.append(V_T)
    
    # print(V_Ts)
  
    
    # Compute cosine angle for each pair of V_Ts
    cosine_angles = []
    for i in range(len(V_Ts)):
        cos_i = []
        # for j in range(len(V_Ts)):
        for j in range(i + 1, len(V_Ts)):
            cosine_angle = torch.dot(V_Ts[i], V_Ts[j]) / (torch.norm(V_Ts[i]) * torch.norm(V_Ts[j]))
            # cosine_angles.append(cosine_angle)
            cos_i.append(cosine_angle)
        cosine_angles.append(cos_i)
        
    # 导出对角线上的cosine angle
    _diagonal_cosine_angles = []
    for i in range(1, len(V_Ts)):
        cosine_angle = torch.dot(V_Ts[i-1], V_Ts[i]) / (torch.norm(V_Ts[i-1]) * torch.norm(V_Ts[i]))
        _diagonal_cosine_angles.append(cosine_angle)
        
    
    # export to txt, split by ',' and reserve 2位小数
    with open('cosine_angles_stepsize{}.txt'.format(STEP_SIZE), 'w') as f:
        # f.write(','.join([str(cosine_angle.item()) for cosine_angle in _diagonal_cosine_angles]))
        f.write(','.join([str(round(cosine_angle.item(), 2)) for cosine_angle in _diagonal_cosine_angles]))
    # exit()
    
    
    
    # print(cosine_angles)
    
    # visualize cosine_angles
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()
    plt.figure(figsize=(12, 12))
    # adjust the 分辨率
    # plt.rcParams['figure.dpi'] = 300
    # title
    plt.title('Cosine Angles between V_Ts, step size = {}'.format(STEP_SIZE))

    data_array = np.zeros((len(cosine_angles), len(cosine_angles)))
    for i in range(len(cosine_angles)):
        for j in range(len(cosine_angles[i])):
            data_array[i][i+j+1] = cosine_angles[i][j]

    sns.heatmap(data_array, annot=False)
    plt.show()
    
    
    print('Done')
    
    
    
    