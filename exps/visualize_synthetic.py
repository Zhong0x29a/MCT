import argparse
import copy
import os

import math
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm

from reparam_module import ReparamModule
from utils import get_dataset, ParamDiffAug, get_network, epoch, TensorDataset


def eval_theta(_theta, testloader, net, args):
    net.flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in _theta], 0))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    test_loss, test_acc = epoch("test", dataloader=testloader, net=net, optimizer=None,
                                criterion=criterion, args=args, aug=False)
    
    return test_loss, test_acc

def get_pca_points(_trajectory):
    # return 2-d points, list:x, list:y
    pca = PCA(n_components=2)
    theta_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in _trajectory]
    pca.fit(theta_points)
    
    # Transform the points to the first 2 principal components
    transformed_points = pca.transform(theta_points)
    
    return transformed_points[:, 0], transformed_points[:, 1]


def plot_trajs_acc_combined(sets, sets_name, filename):
    fig = go.Figure()
    # colorscales = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges']
    colorscales = [
        [[0, 'rgb(0, 0, 255)'], [0.5, 'rgb(100, 100, 255)'], [1, 'rgb(200, 200, 255)']],  # Blues
        [[0, 'rgb(255, 0, 0)'], [0.5, 'rgb(255, 100, 100)'], [1, 'rgb(255, 200, 200)']],  # Reds
        [[0, 'rgb(128, 0, 128)'], [0.5, 'rgb(178, 100, 178)'], [1, 'rgb(238, 200, 238)']],  # Purples
        [[0, 'rgb(255, 165, 0)'], [0.5, 'rgb(255, 195, 100)'], [1, 'rgb(255, 225, 200)']],  # Oranges
        [[0, 'rgb(0, 255, 0)'], [0.5, 'rgb(100, 255, 100)'], [1, 'rgb(200, 255, 200)']],  # Greens
    ]
    for idx, set_i in enumerate(sets):
        fig.add_trace(go.Scatter3d(
            x=set_i["x"],
            y=set_i["y"],
            z=set_i["z"],
            mode='markers',
            marker=dict(
                size=5,
                color=set_i["z"],
                colorscale=colorscales[idx % len(colorscales)],
                opacity=0.8
            ),
            name="Set {}".format(sets_name[idx] if sets_name is not None else idx)
        ))

    # Save the plot as an HTML file
    pio.write_html(fig, filename)
   
# def get_acc_traj()

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    
    if True:
        ''' load the buffer '''
        expert_dir = args.buffer
        expert_files = []
        n = 0
        
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        
        ''' get real data info. '''
        buffer = torch.load(expert_files[0])
        traj_real = buffer[0]
        acc_real = []
        
        tool_net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        tool_net = ReparamModule(tool_net).eval()
        
        for theta in tqdm(traj_real, total=len(traj_real), desc="Evaluating"):
            loss, acc = eval_theta(theta, testloader, tool_net, args)
            acc_real.append(acc)
    
    if True:
        ''' load the buffer '''
        expert_dir = args.buffer
        expert_files = []
        n = 0
        
        while os.path.exists(os.path.join(expert_dir, "convexified_replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "convexified_replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        
        ''' get real data info. '''
        buffer = torch.load(expert_files[0])
        traj_real_2 = buffer[0]
        acc_real_2 = []
        
        tool_net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        tool_net = ReparamModule(tool_net).eval()
        
        for theta in tqdm(traj_real_2, total=len(traj_real_2), desc="Evaluating"):
            loss, acc = eval_theta(theta, testloader, tool_net, args)
            acc_real_2.append(acc)

    
    ''' get synthetic data info. '''
    synthetic_img = torch.load("/root/zhongwenliang/DC-MTT/logged_files/CIFAR10/original/images_ipc-10_best.pt")
    synthetic_lab = torch.load("/root/zhongwenliang/DC-MTT/logged_files/CIFAR10/original/labels_ipc-10_best.pt")
    
    # sftp://root@211.87.232.86:20025/root/zhongwenliang/DC-MTT/logged_files/CIFAR10/convexified/1/labels_ipc-10_best.pt
    synthetic_img_2 = torch.load("/root/zhongwenliang/DC-MTT/logged_files/CIFAR10/convexified/1/images_ipc-10_best.pt")
    synthetic_lab_2 = torch.load("/root/zhongwenliang/DC-MTT/logged_files/CIFAR10/convexified/1/labels_ipc-10_best.pt")
    
    # mtt
    if True:
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        
        for param, real_param in zip(net.parameters(), traj_real[0]):
            param.data = real_param.data
        
        images_train = synthetic_img.to(args.device)
        labels_train = synthetic_lab.to(args.device)
        lr = float(0.011)
        Epoch = int(1000)
        lr_schedule = [Epoch // 2 + 1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
        criterion = nn.CrossEntropyLoss().to(args.device)
        
        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
        
        traj_synt = []
        acc_synt = []
        net.train()
        # train for 1000 epochs, save every 20 epochs
        for ep in tqdm(range(Epoch + 1), desc="Training synthetic"):
            if ep % 20 == 0:
                theta = [p.detach().clone().cpu() for p in net.parameters()]
                traj_synt.append(theta)
                _, acc = eval_theta(theta, testloader, tool_net, args)
                acc_synt.append(acc)
            
            loss_train, acc_train = epoch("train", dataloader=trainloader, net=net, optimizer=optimizer,
                                          criterion=criterion, args=args, aug=args.dsa)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # convexied
    if True:
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        
        for param, real_param in zip(net.parameters(), traj_real[0]):
            param.data = real_param.data
        
        synthetic_img = synthetic_img_2
        synthetic_lab = synthetic_lab_2
        
        images_train = synthetic_img.to(args.device)
        labels_train = synthetic_lab.to(args.device)
        lr = float(0.011)
        Epoch = int(1000)
        lr_schedule = [Epoch // 2 + 1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
        criterion = nn.CrossEntropyLoss().to(args.device)
        
        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
        
        traj_synt_2 = []
        acc_synt_2 = []
        net.train()
        # train for 1000 epochs, save every 20 epochs
        for ep in tqdm(range(Epoch + 1), desc="Training synthetic"):
            if ep % 20 == 0:
                theta = [p.detach().clone().cpu() for p in net.parameters()]
                traj_synt_2.append(theta)
                _, acc = eval_theta(theta, testloader, tool_net, args)
                acc_synt_2.append(acc)
            
            loss_train, acc_train = epoch("train", dataloader=trainloader, net=net, optimizer=optimizer,
                                            criterion=criterion, args=args, aug=args.dsa)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            
    # visualize the trajectory
    z_real = [1 - acc for acc in acc_real]
    z_real_2 = [1 - acc for acc in acc_real_2]
    z_synt = [1 - acc for acc in acc_synt]
    z_synt_2 = [1 - acc for acc in acc_synt_2]
    
    # pca_all = get_pca_points(traj_real + traj_synt + traj_synt_2)
    pca_all = get_pca_points(traj_real + traj_real_2 + traj_synt + traj_synt_2)
    pca_xs, pca_ys = pca_all
    
    # x_real, y_real = pca_xs[:len(traj_real)], pca_ys[:len(traj_real)]
    # x_synt, y_synt = pca_xs[len(traj_real):len(traj_real) + len(traj_synt)], pca_ys[len(traj_real):len(traj_real) + len(traj_synt)]
    # x_synt_2, y_synt_2 = pca_xs[len(traj_real) + len(traj_synt):], pca_ys[len(traj_real) + len(traj_synt):]
    x_real, y_real = pca_xs[:len(traj_real)], pca_ys[:len(traj_real)]
    x_real_2, y_real_2 = pca_xs[len(traj_real):len(traj_real) + len(traj_real_2)], pca_ys[len(traj_real):len(traj_real) + len(traj_real_2)]
    x_synt, y_synt = pca_xs[len(traj_real) + len(traj_real_2):len(traj_real) + len(traj_real_2) + len(traj_synt)], pca_ys[len(traj_real) + len(traj_real_2):len(traj_real) + len(traj_real_2) + len(traj_synt)]
    x_synt_2, y_synt_2 = pca_xs[len(traj_real) + len(traj_real_2) + len(traj_synt):], pca_ys[len(traj_real) + len(traj_real_2) + len(traj_synt):]
    
    real_set = {"x": x_real, "y": y_real, "z": z_real}
    real_set_2 = {"x": x_real_2, "y": y_real_2, "z": z_real_2}
    synt_set = {"x": x_synt, "y": y_synt, "z": z_synt}
    synt_set_2 = {"x": x_synt_2, "y": y_synt_2, "z": z_synt_2}
    
    
    # plot_trajs_acc_combined([real_set, synt_set], ['real', 'synthetic'],
    #                         "combined_plot_{}.html".format("MTT"))
    plot_trajs_acc_combined([real_set, real_set_2, synt_set, synt_set_2],
                            ['real', 'real_2', 'synthetic', 'synthetic_2'],
                            "combined_plot_{}.html".format("All"))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for Convexify')
    
    # buffer path
    parser.add_argument('--buffer', type=str, default="../buffers/CIFAR10_NO_ZCA/ConvNet/",
                        help='Path to the buffer file')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=2048, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=2048, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='../buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    
    # alpha
    parser.add_argument("--alpha", type=float, default=0.3, help="alpha for interpolation")
    
    args = parser.parse_args()
    main(args)
    # parser.add_argument('--max_files', type=int, help='Maximum number of files to load')
    # parser.add_argument('--max_experts', type=int, help='Maximum number of experts to load')
    #
    # args = parser.parse_args()
    # main(args)
