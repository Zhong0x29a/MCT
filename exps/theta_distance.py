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

from exps.trajectory_compression import CompressedTrajectoryWithInterpolation
from reparam_module import ReparamModule
from utils import get_dataset, ParamDiffAug, get_network, epoch


def minus_through_listed_tensors(A, B):
    return [a - b for a, b in zip(A, B)]


def mean_two_through_listed_tensors(A, B):
    return [torch.div(torch.add(a, b), 2) for a, b in zip(A, B)]


def convexify_trajectory(_trajectroy):
    delta_theta = minus_through_listed_tensors(_trajectroy[-1], _trajectroy[0])
    
    normalized_delta_theta = []
    for layer in delta_theta:
        normed_layer = torch.norm(layer, p=2)
        if normed_layer == 0:
            normed_layer = 1
        normalized_delta_theta.append(torch.div(layer, normed_layer))
    
    target_trajectroy = [_trajectroy[0]]
    # calc traget theta_i
    for i in range(1, len(_trajectroy)):
        delta_theta_0Toi = minus_through_listed_tensors(_trajectroy[i], _trajectroy[0])
        
        vec_0Toi = [torch.mul(torch.norm(layer_0Toi, p=2), normed_layer) for layer_0Toi, normed_layer in
                    zip(delta_theta_0Toi, normalized_delta_theta)]
        
        target_theta = [torch.add(layer, _trajectroy[0][idx]) for idx, layer in enumerate(vec_0Toi)]
        
        # determine nan
        for layer in target_theta:
            if torch.isnan(layer).any():
                print("Nan location: ", torch.isnan(layer))
                
                
        target_trajectroy.append(target_theta)
    
    return target_trajectroy


def convexify_a_trajectory(_trajectory):
    delta_theta = minus_through_listed_tensors(_trajectory[-1], _trajectory[0])
    
    accumulated_delta = [torch.tensor(0.) for _ in range(len(delta_theta))]
    for i in range(1, len(_trajectory)):
        _delta_theta = minus_through_listed_tensors(_trajectory[i], _trajectory[i - 1])
        accumulated_delta = [torch.add(layer, torch.norm(delta, p=2)) for layer, delta in
                             zip(accumulated_delta, _delta_theta)]
    
    _traget_trajectory = [_trajectory[0]]
    current_point = _trajectory[0]
    for i in range(1, len(_trajectory)):
        delta_i = minus_through_listed_tensors(_trajectory[i], _trajectory[i - 1])
        norm_delta_i = [torch.norm(layer, p=2) for layer in delta_i]
        
        take_up = [torch.div(norm, _sum) for _sum, norm in zip(accumulated_delta, norm_delta_i)]
        # todo: avoid zero division, if _sum is zero, then take_up = 1
        take_up = [torch.tensor(1.) if _sum == 0 else take for _sum, take in zip(accumulated_delta, take_up)]
        
        # vec_next = delta_theta * take_up
        vec_next = [torch.mul(layer, take) for layer, take in zip(delta_theta, take_up)]
        
        current_point = [torch.add(layer, vec) for layer, vec in zip(current_point, vec_next)]
        _traget_trajectory.append(current_point)
    
    return _traget_trajectory


"""
interpolate some points between start and end point
"""
def convexify_trajectory_with_interpolation(_trajectory, idxs=(15, 30)):
    _trajectory = copy.deepcopy(_trajectory)
    _trajectory[50] = mean_two_through_listed_tensors(_trajectory[50], _trajectory[49])
    _target_trajectory = []
    
    # smooth the waypoints by their neighbors
    for i in range(len(idxs)):
        tmp = mean_two_through_listed_tensors(_trajectory[idxs[i]+1], _trajectory[idxs[i]-1])
        _trajectory[idxs[i]] = mean_two_through_listed_tensors(_trajectory[idxs[i]], tmp)
        
    # smooth the final point
    tmp = mean_two_through_listed_tensors(_trajectory[-3], _trajectory[-2])
    _trajectory[-1] = mean_two_through_listed_tensors(_trajectory[-1], tmp)
    
    _target_trajectory.extend(convexify_a_trajectory(_trajectory[:idxs[0] + 1]))
    for i in range(len(idxs) - 1):
        _target_trajectory.extend(convexify_a_trajectory(_trajectory[idxs[i]:idxs[i + 1] + 1])[1:])
    _target_trajectory.extend(convexify_a_trajectory(_trajectory[idxs[-1]:])[1:])
    
    return _target_trajectory


def eval_theta(_theta, testloader, net, args):
    net.flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in _theta], 0))
    
    # with torch.no_grad():
    criterion = nn.CrossEntropyLoss().to(args.device)
    test_loss, test_acc = epoch("test", dataloader=testloader, net=net, optimizer=None,
                                criterion=criterion, args=args, aug=False)
    
    return test_loss, test_acc
 
def plot_distance(trajectory, name=None):
    delt = []
    for i in range(len(trajectory)-1):
        minus = minus_through_listed_tensors(trajectory[i+1], trajectory[i])
        delt.append([torch.norm(layer, p=2) for layer in minus])
        
    import matplotlib.pyplot as plt
    # title
    plt.title("Delta theta curve"+(" "+name if name is not None else ""))
    for i in range(len(delt[0])):
        plt.plot([d[i] for d in delt], label="Layer {}".format(i))
    plt.legend()
    
    plt.show()
    

def add_noised_to_trajectory(_trajectory):
    _trajectory = copy.deepcopy(_trajectory)
    for idx, theta in enumerate(_trajectory):
        # add small noise to the starting parameters
        for p in theta:
            # if p is 1. or zero, don't add noise
            if torch.sum(torch.abs(p)) == 0:
                continue
            if torch.sum(torch.abs(p)) == 1.:
                continue
            
            # set the epsilon to be 1% of the parameter value
            epsilons = torch.abs(p) * 0.05
            noise = torch.randn_like(p) * epsilons
            p += noise
    
    return _trajectory
    

def visualize_trajectory(_trajectory, name="traj"):
    import plotly.graph_objects as go
    import plotly.io as pio
    pca = PCA(n_components=3)
    theta_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in _trajectory]
    pca.fit(theta_points)

    # Transform the points to the first 3 principal components
    transformed_points = pca.transform(theta_points)
    
    colors = np.linspace(0.1, 1, len(transformed_points))
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=transformed_points[:, 0],
        y=transformed_points[:, 1],
        z=transformed_points[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])
    
    # Add a special marker for the starting point
    fig.add_trace(go.Scatter3d(
        x=[transformed_points[0, 0]],
        y=[transformed_points[0, 1]],
        z=[transformed_points[0, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='pink',  # set color to red
            opacity=1
        ),
        name='Starting Point'
    ))
    
    # Save the plot as an HTML file
    pio.write_html(fig, "plot_{}.html".format(name))

def visualize_trajectory_tsne(_trajectory, name="traj"):
    from sklearn.manifold import TSNE
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np
    
    # Perform t-SNE
    tsne = TSNE(n_components=3)
    theta_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in _trajectory]
    transformed_points = tsne.fit_transform(theta_points)

    # Create a color array that changes from light to dark
    colors = np.linspace(0.1, 1, len(transformed_points))

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=transformed_points[:, 0],
        y=transformed_points[:, 1],
        z=transformed_points[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

    # Save the plot as an HTML file
    pio.write_html(fig, "tsne_plot_{}.html".format(name))

def visualize_trajectory_isomap(_trajectory, name="traj"):
    from sklearn.manifold import Isomap
    import numpy as np
    # Perform Isomap
    isomap = Isomap(n_components=3)
    theta_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in _trajectory]
    transformed_points = isomap.fit_transform(theta_points)

    # Create a color array that changes from light to dark
    colors = np.linspace(0.1, 1, len(transformed_points))

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=transformed_points[:, 0],
        y=transformed_points[:, 1],
        z=transformed_points[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

    # Save the plot as an HTML file
    pio.write_html(fig, "isomap_plot_{}.html".format(name))


def get_pca_points(_trajectory):
    # return 2-d points, list:x, list:y
    pca = PCA(n_components=2)
    theta_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in _trajectory]
    pca.fit(theta_points)
    
    # Transform the points to the first 2 principal components
    transformed_points = pca.transform(theta_points)
    
    return transformed_points[:, 0], transformed_points[:, 1]

def export_3d_points(_x, _y, _z, filename=""):
    fig = go.Figure(data=[go.Scatter3d(
        x=_x,
        y=_y,
        z=_z,
        mode='markers',
        marker=dict(
            size=6,
            color=_z,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])
    
    # Save the plot as an HTML file
    pio.write_html(fig, filename)
    
    
def plot_trajs_acc(buffer, testloader, net, args, filename):
    traj_subset = [theta for trj in buffer for theta in trj]
    point2d = get_pca_points(traj_subset)
    accs = []
    # use tqdm()
    for idx, theta in tqdm(enumerate(traj_subset), total=len(traj_subset), desc="Evaluating"):
        loss, acc = eval_theta(theta, testloader, net, args)
        accs.append(acc)
    accs = [1 - acc for acc in accs]
    export_3d_points(point2d[0], point2d[1], accs, filename)

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    
    net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
    net = ReparamModule(net).eval()
    
    ''' load the buffer '''
    expert_dir = args.buffer
    expert_files = []
    n = 0
    
    # todo: temp
    n=1
    
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    
    # conv_expert_files = []
    # n = 0
    # while os.path.exists(os.path.join(expert_dir, "convexified_replay_buffer_{}.pt".format(n))):
    #     conv_expert_files.append(os.path.join(expert_dir, "convexified_replay_buffer_{}.pt".format(n)))
    #     n += 1
    if n == 0:
        raise AssertionError("No convexified buffers detected at {}".format(expert_dir))
    
    for file_idx, file in enumerate(expert_files):
        print("Processing Buffer {}".format(file_idx))
        
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        
        # buffer_conv = torch.load(conv_expert_files[file_idx])
        
        print("{} Buffers loaded".format(len(buffer)))
        
        acc_hat_list_list = []
        acc_ori_list_list = []
        
      
        for traj_i, trajectory in enumerate(buffer):
            
            acc_hat_list = []
            acc_hat_v2_list = []
            acc_ori_list = []
            
            trajectory = buffer[traj_i]
            # convexified_trajectory = buffer_conv[traj_i]
            convexified_trajectory = convexify_a_trajectory(trajectory)
            convexified_trajectory_v2 = convexify_trajectory_with_interpolation(trajectory, idxs=(15, 30))
            
            for idx, (theta_hat, theta_hat_v2, theta) in enumerate(zip(convexified_trajectory, convexified_trajectory_v2,
                                                                       trajectory)):
                
                loss_hat, acc_hat = eval_theta(theta_hat, testloader, net, args)
                loss_hat_v2, acc_hat_v2 = eval_theta(theta_hat_v2, testloader, net, args)
                test_loss, test_acc = eval_theta(theta, testloader, net, args)
                
                acc_hat_list.append(acc_hat)
                acc_hat_v2_list.append(acc_hat_v2)
                acc_ori_list.append(test_acc)
                
                print("Traj: {}\tEpoch: {}\tConvex Acc: {},\t{}\tOri Acc: {}".format(traj_i, idx, acc_hat,
                                                                                      acc_hat_v2, test_acc))

            
            # plot 3-d pca with
            pca = PCA(n_components=2)
            theta_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in trajectory]
            theta_points += [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in convexified_trajectory]
            
            pca.fit(theta_points)
            transformed_points_ori = pca.transform(theta_points[:len(trajectory)])
            transformed_points_hat = pca.transform(theta_points[len(trajectory):])
            
            colors_ori = np.linspace(1, 1, len(transformed_points_ori))
            colors_hat = np.linspace(1, 1, len(transformed_points_hat))
            
            # Create a 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=transformed_points_ori[:, 0],
                y=transformed_points_ori[:, 1],
                z=[(1 - acc) for acc in acc_ori_list],  # set z to an array/list of accuracy values
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors_ori,  # set color to an array/list of desired values
                    colorscale='Greens',  # choose a colorscale
                    opacity=0.8
                )
            )])
            fig.add_trace(go.Scatter3d(
                x=transformed_points_hat[:, 0],
                y=transformed_points_hat[:, 1],
                z=[(1 - acc) for acc in acc_hat_list],  # set z to an array/list of accuracy values
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors_hat,  # set color to an array/list of desired values
                    colorscale='Blues',  # choose a colorscale
                    opacity=0.8
                )
            ))
            pio.write_html(fig, "plot_{}.html".format(traj_i))
            
            
            # plot the acc_hat and test_acc curve comparing with the epoch
            import matplotlib.pyplot as plt
            plt.figure(dpi=340)
            # adjust the dots per inch to make the plot larger

            plt.plot(acc_hat_list, label='Convex Acc')
            plt.plot(acc_hat_v2_list, label='Convex Acc v2')
            plt.plot(acc_ori_list, label='Ori Acc')
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.title("Comparison of Convex Acc and Ori Acc")
            plt.legend()

            plt.show()
            # exit()
            # save each to a file
            # plt.savefig(os.path.join('../plots/', "traj_acc_compare_{}.png".format(traj_i)))
            
            acc_hat_list_list.append(acc_hat_list)
            acc_ori_list_list.append(acc_ori_list)
        
        # calc the mean and std of the acc_hat and acc_ori
        acc_hat_list_mean = [sum([acc_list[i] for acc_list in acc_hat_list_list]) / len(acc_hat_list_list) for i in range(len(acc_hat_list_list[0]))]
        acc_ori_list_mean = [sum([acc_list[i] for acc_list in acc_ori_list_list]) / len(acc_ori_list_list) for i in range(len(acc_ori_list_list[0]))]
        
        acc_hat_list_std = [sum([(acc_list[i] - acc_hat_list_mean[i]) ** 2 for acc_list in acc_hat_list_list]) / len(acc_hat_list_list) for i in range(len(acc_hat_list_list[0]))]
        acc_ori_list_std = [sum([(acc_list[i] - acc_ori_list_mean[i]) ** 2 for acc_list in acc_ori_list_list]) / len(acc_ori_list_list) for i in range(len(acc_ori_list_list[0]))]
        
        # plot the acc_hat and test_acc curve comparing with the epoch
        import matplotlib.pyplot as plt
        plt.figure(dpi=300)
        # adjust the dots per inch to make the plot larger
        x=range(len(acc_hat_list_mean))
        # plt.errorbar(x, acc_hat_list_mean, yerr=acc_hat_list_std, label='Convex Acc')
        # plt.errorbar(x, acc_ori_list_mean, yerr=acc_ori_list_std, label='Ori Acc')
        # change to fill_between
        plt.fill_between(x, [mean - std for mean, std in zip(acc_hat_list_mean, acc_hat_list_std)], [mean + std for mean, std in zip(acc_hat_list_mean, acc_hat_list_std)], alpha=0.1, label='Convex Acc')
        plt.fill_between(x, [mean - std for mean, std in zip(acc_ori_list_mean, acc_ori_list_std)], [mean + std for mean, std in zip(acc_ori_list_mean, acc_ori_list_std)], alpha=0.1, label='Ori Acc')
        
        plt.plot(acc_hat_list_mean, label='Convex Acc')
        plt.plot(acc_ori_list_mean, label='Ori Acc')
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.title('Comparison of Convex Acc and Ori Acc')
        plt.legend()
        # plt.show()
        # save each to a file
        plt.savefig(os.path.join('../plots/', "traj_acc_compare_mean_std_{}.png".format(file_idx)))
        
        exit()
        

''' plot 3-d html '''
def plot_3traj_combined(n_po=51):
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    
    net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
    net = ReparamModule(net).eval()
    
    
    path_ori = '/root/zhongwenliang/DC-MTT/buffers/CIFAR10_NO_ZCA/ConvNet/replay_buffer_0.pt'
    
    path_v1_waypoint = '/root/zhongwenliang/DC-MTT/buffers/CIFAR10_NO_ZCA/ConvNet/waypoint_buffer_0.pt'
    path_v1_step = '/root/zhongwenliang/DC-MTT/buffers/CIFAR10_NO_ZCA/ConvNet/step_buffer_0.pt'
    
    path_v2_waypoint = '/root/zhongwenliang/DC-MTT/buffers/CIFAR10_NO_ZCA/ConvNet/waypoint_buffer_ip4_0.pt'
    path_v2_step = '/root/zhongwenliang/DC-MTT/buffers/CIFAR10_NO_ZCA/ConvNet/step_buffer_ip4_0.pt'
    
    buffer_ori = torch.load(path_ori)
    traj_ori = buffer_ori[0]
    
    traj_v1_waypoint = torch.load(path_v1_waypoint)[0]
    traj_v1_step = torch.load(path_v1_step)[0]
    traj_v1_ct = CompressedTrajectoryWithInterpolation(traj_v1_waypoint, traj_v1_step, 2)
    traj_v1 = []
    
    traj_v2_waypoint = torch.load(path_v2_waypoint)[0]
    traj_v2_step = torch.load(path_v2_step)[0]
    traj_v2_ct = CompressedTrajectoryWithInterpolation(traj_v2_waypoint, traj_v2_step, 4)
    traj_v2 = []
    
    acc_ori = []
    acc_v1 = []
    acc_v2 = []
    
    # read from file
    with open("acc_both.txt", "r") as f:
        lines = f.readlines()
        acc_ori = eval(lines[0].split(":")[1])
        acc_v1 = eval(lines[1].split(":")[1])
        acc_v2 = eval(lines[2].split(":")[1])
    
    for i in range(n_po):
        # acc_ori.append(1-eval_theta(traj_ori[i], testloader, net, args)[1])
        
        theta_v1 = traj_v1_ct.get_point_n(i)
        theta_v2 = traj_v2_ct.get_point_n(i)
        traj_v1.append(theta_v1)
        traj_v2.append(theta_v2)
        # acc_v1.append(1-eval_theta(theta_v1, testloader, net, args)[1])
        # acc_v2.append(1-eval_theta(theta_v2, testloader, net, args)[1])
    
    print(acc_ori)
    print(acc_v1)
    print(acc_v2)
    # save acc
    # with open("acc_both.txt", "w") as f:
    #     f.write("acc_ori: {}\n".format(acc_ori))
    #     f.write("acc_v1: {}\n".format(acc_v1))
    #     f.write("acc_v2: {}\n".format(acc_v2))
    #     print('saved to acc_both.txt')
    #     exit()

    pca = PCA(n_components=2)
    all_points = [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in traj_ori]+\
                    [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in traj_v1]+\
                    [torch.cat([p.reshape(-1) for p in theta], 0).detach().numpy() for theta in traj_v2]

    pca.fit(all_points)
    
    transformed_points_ori = pca.transform(all_points[:len(traj_ori)])
    transformed_points_v1 = pca.transform(all_points[len(traj_ori):len(traj_ori)+len(traj_v1)])
    transformed_points_v2 = pca.transform(all_points[len(traj_ori)+len(traj_v1):])
    
    xs_ori = transformed_points_ori[:, 0]
    ys_ori = transformed_points_ori[:, 1]
    xs_v1 = transformed_points_v1[:, 0]
    ys_v1 = transformed_points_v1[:, 1]
    xs_v2 = transformed_points_v2[:, 0]
    ys_v2 = transformed_points_v2[:, 1]
        
    # green for ori, blue for v1, red for v2
    # plot all points into single html, (x,y,acc)
    mark_size = 5
    fig = go.Figure(data=[go.Scatter3d(
        x=xs_ori,
        y=ys_ori,
        z=acc_ori,
        mode='markers',
        marker=dict(
            size=mark_size,
            color='#fb7299',  # set color to an array/list of desired values
            # colorscale='#8bcc90',  # choose a colorscale
            opacity=0.8
        )
    )])
    fig.add_trace(go.Scatter3d(
        x=xs_v1,
        y=ys_v1,
        z=acc_v1,
        mode='markers',
        marker=dict(
            size=mark_size,
            color='#83badc',  # set color to an array/list of desired values
            # colorscale='Blue',  # choose a colorscale
            opacity=0.8
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=xs_v2,
        y=ys_v2,
        z=acc_v2,
        mode='markers',
        marker=dict(
            size=mark_size,
            color='#8bcc90',  # set color to an array/list of desired values
            # colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    ))
    
    pio.write_html(fig, "final_____plot_3traj_combined.html")
    print("Done")
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for Convexify')
    
    # buffer path
    parser.add_argument('--buffer', type=str, default="../buffers/CIFAR10_NO_ZCA/ConvNet/",
    # parser.add_argument('--buffer', type=str, default="../buffers/CIFAR100/ConvNet/",
    # parser.add_argument('--buffer', type=str, default="../buffers/Tiny/ConvNet/",
                        help='Path to the buffer file')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
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
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    # alpha
    parser.add_argument("--alpha", type=float, default=0.3, help="alpha for interpolation")
    
    args = parser.parse_args()
    
    
    ###############
    plot_3traj_combined(50)
    
    
    # main(args)
    # parser.add_argument('--max_files', type=int, help='Maximum number of files to load')
    # parser.add_argument('--max_experts', type=int, help='Maximum number of experts to load')
    #
    # args = parser.parse_args()


"""
--dataset=Tiny --data_path=../data/tiny-imagenet-200 --zca
"""