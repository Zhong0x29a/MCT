import os

import argparse
import torch

from convexify import minus_through_listed_tensors
from utils import ParamDiffAug, get_network, get_dataset


class CompressedTrajectory:
    """
    Encapsulated class for compressed trajectory.
    Initialize with start and end point, and a list of weights along the trajectory.
    
    get_point_n: get the desired point along the trajectory through interpolation.
    """
    def __init__(self, start_point, end_point, weight_list):
        self.start_point = start_point
        self.end_point = end_point
        self.weight_list = weight_list
        
    def get_waypoint(self, weight):
        """
        interpolate some points between start and end point
        usage e.g.: theta = get_waypoint(traj_info[0][0], traj_info[0][1], weight_info[0][i])
        :param weight: list of accumulated weights (tensor) for each layer
        :return:
        """
        # check if nan
        for w in weight:
            if torch.isnan(w).any():
                raise ValueError("NaN detected in weight")
        for s, e in zip(self.start_point, self.end_point):
            if torch.isnan(s).any() or torch.isnan(e).any():
                raise ValueError("NaN detected in start or end point")
        return [torch.lerp(start, end, _wei) for start, end, _wei in zip(self.start_point, self.end_point, weight)]
    
    @staticmethod
    def interpolate_weight(start_weight, end_weight, alpha):
        """
        interpolate some points between start and end point
        usage e.g.: theta = get_waypoint(traj_info[0][0], traj_info[0][1], weight_info[0][i])
        :param start_weight: list of weights (tensor) for each layer
        :param end_weight: list of weights (tensor) for each layer
        :param alpha: float
        :return:
        """
        return [torch.lerp(start, end, alpha) for start, end in zip(start_weight, end_weight)]
        
    def get_point_n(self, n):
        """
        interpolate desired points between start and end point
        e.g. get_point_n(traj_info[t_i][0], traj_info[t_i][1], weight_info[t_i], 49.5)
        :param n: target point location, float, between the start and end point
        :return:
        """
        assert 0 <= n <= len(self.weight_list) - 1
        low_n = int(n)
        high_n = int(low_n + 1) if low_n < len(self.weight_list) - 1 else int(len(self.weight_list) - 1)
        alpha = n - low_n
        new_weight = self.interpolate_weight(self.weight_list[low_n], self.weight_list[high_n], alpha)
        return self.get_waypoint(new_weight)

class CompressedTrajectoryWithInterpolation:
    """
    Encapsulated class for compressed trajectory with interpolation.
    
    usage:
    waypoints from file: waypoint_list = torch.load("waypoint_buffer_0.pt")
    step_size from file: step_size_list = torch.load("step_buffer_0.pt")
    num_interpolation from file name: ip3, ip4, ... or none for ip2
    """
    def __init__(self, waypoints_list, step_size_list, num_interpolation):
        self.waypoints = waypoints_list
        self.step_size_list = step_size_list
        self.num_interpolation = num_interpolation
        self.waypoints_idx = [0]
        if num_interpolation > 2:
            for i in range(num_interpolation-1):
                self.waypoints_idx.append(len(step_size_list[i]) + self.waypoints_idx[-1] - 1)
            
        else:
            self.waypoints_idx.append(len(step_size_list) - 1)
            
    @staticmethod
    def get_waypoint(start_point, end_point, weight):
        """
        interpolate some points between start and end point
        usage e.g.: theta = get_waypoint(traj_info[0][0], traj_info[0][1], weight_info[0][i])
        :param start_point: list of weights (tensor) for each layer
        :param end_point: list of weights (tensor) for each layer
        :param weight: list of accumulated weights (tensor) for each layer
        :return:
        """
        # check if nan
        for w in weight:
            if torch.isnan(w).any():
                raise ValueError("NaN detected in weight")
        for s, e in zip(start_point, end_point):
            if torch.isnan(s).any() or torch.isnan(e).any():
                raise ValueError("NaN detected in start or end point")
        return [torch.lerp(start, end, _wei) for start, end, _wei in zip(start_point, end_point, weight)]
        
    @staticmethod
    def interpolate_weight(start_weight, end_weight, alpha):
        """
        interpolate some points between start and end point
        usage e.g.: theta = get_waypoint(traj_info[0][0], traj_info[0][1], weight_info[0][i])
        :param start_weight: list of weights (tensor) for each layer
        :param end_weight: list of weights (tensor) for each layer
        :param alpha: float
        :return:
        """
        return [torch.lerp(start, end, alpha) for start, end in zip(start_weight, end_weight)]
        
    def get_point_n(self, n):
        assert 0 <= n <= self.waypoints_idx[-1]
        alpha = n - int(n)
        # get range of n
        if self.num_interpolation > 2:
            for idx in range(len(self.waypoints_idx) - 1):
                if self.waypoints_idx[idx] <= n < self.waypoints_idx[idx + 1]:
                    low_n = int(n) - self.waypoints_idx[idx]
                    high_n = int(low_n + 1)
                    high_n = min(high_n, self.waypoints_idx[-1])
                    
                    weight_low = self.step_size_list[idx][low_n]
                    weight_high = self.step_size_list[idx][high_n]
                    new_weight = self.interpolate_weight(weight_low, weight_high, alpha)
                    
                    return self.get_waypoint(self.waypoints[idx], self.waypoints[idx + 1], new_weight)
        else:
            low_n = int(n)
            high_n = int(low_n + 1) if low_n < len(self.step_size_list) - 1 else int(len(self.step_size_list) - 1)
            weight_low = self.step_size_list[low_n]
            weight_high = self.step_size_list[high_n]
            new_weight = self.interpolate_weight(weight_low, weight_high, alpha)
            return self.get_waypoint(self.waypoints[0], self.waypoints[1], new_weight)
                

def mean_two_through_listed_tensors(A, B):
    return [torch.div(torch.add(a, b), 2) for a, b in zip(A, B)]

def get_norm2_between_nodes(_trajectory):
    """
    get the norm2 of the difference between two adjacent nodes in the trajectory
    example:
    input traj=[theta_0, theta_1, ..., theta_50].
    then the output will be:
    [norm(theta_1 - theta_0), norm(theta_2 - theta_1), ..., norm(theta_50 - theta_49)],
    where the theta_i is a list of tensors representing the weights of each layer.
    :param _trajectory:
    :return:
    """
    traj_norm2 = []
    for i in range(1, len(_trajectory)):
        diff = minus_through_listed_tensors(_trajectory[i], _trajectory[i - 1])
        norm2 = [torch.norm(layer, p=2) for layer in diff]
        traj_norm2.append(norm2)
    return traj_norm2


def get_step_and_waypoint(trajectory):
    """
    Get the step size and waypoint list for the trajectory.
    :param trajectory: list of network weights. e.g. [theta_0, theta_1, ..., theta_n]
    :return: step_size_list, waypoint_list
        step_size_list:
        [         (layer 0)       (layer 1)   ...     (layer n)
        (theta_0) [tensor(0.),    tensor(0.), ...,    tensor(0.)],
        (theta_1) [tensor(0.029), tensor(0.0034), ...,tensor(0.0350)],
        ...
        (theta_n) [tensor(1.0),   tensor(1.0), ...,   tensor(1.0)]
        ]
        waypoint_list: [theta_0, theta_n]
    """
    num_points = len(trajectory)
    delta_theta_list = get_norm2_between_nodes(trajectory)
    
    layer_theta_list = []
    for layer_idx in range(len(trajectory[0])):
        sum_delta_norm = torch.sum(torch.stack([delta_theta_list[i][layer_idx] for i in range(len(delta_theta_list))]))
        sum_delta_norm = sum_delta_norm if sum_delta_norm != 0 else 1.
        layer_i_accumulated_list = []
        for point_idx in range(num_points):
            # accumulated_sum = sum([delta_theta_list[i][layer_idx] for i in range(point_idx)] + [torch.tensor(0.)])
            accumulated_sum = torch.sum(torch.stack([delta_theta_list[i][layer_idx] for i in range(point_idx)] + [torch.tensor(0.)]))
            layer_i_accumulated_list.append(accumulated_sum / sum_delta_norm)
        layer_theta_list.append(layer_i_accumulated_list)
        
    # check if nan
    for layer in layer_theta_list:
        for w in layer:
            if torch.isnan(w).any():
                raise ValueError("NaN detected in weight")
        
    step_size_list = []
    for i in range(num_points):
        step_size_list.append([layer[i] for layer in layer_theta_list])
    
    waypoint_list = [trajectory[0], trajectory[-1]]
    
    return step_size_list, waypoint_list
    


def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100", "Tiny"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("dataset: {}, model: {}".format(args.dataset, args.model))
    
    expert_dir = save_dir
    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))

    print("buffer files: ", expert_files)
    # delta_theta = []
    for file_idx, file in enumerate(expert_files):
        buffer = torch.load(file)
        
        step_size_list = []
        waypoint_list = []
        
        for traj_idx, trajectory in enumerate(buffer):
            assert args.num_interpolation >= 2, "num_interpolation should be at least 2"
            if args.num_interpolation > 2:
                waypoints_idx = [round(i) for i in range(0, len(trajectory), len(trajectory) // (args.num_interpolation - 1))]
                if waypoints_idx[-1] != len(trajectory) - 1:
                    waypoints_idx += [len(trajectory) - 1]
                    
                # waypoints_idx = [0, 15, 30, 50]
                # waypoints_idx = [0, 50]

                step_i_piece_list = []
                for i in range(len(waypoints_idx) - 1):
                    step_size, _ = get_step_and_waypoint(trajectory[waypoints_idx[i]:waypoints_idx[i+1]+1])
                    step_i_piece_list.append(step_size)
                    
                step_size_list.append(step_i_piece_list)
                
                # smooth the waypoints
                if len(waypoints_idx[1:-1]) > 0:
                    for idx in waypoints_idx[1:-1]:
                        trajectory[idx] = mean_two_through_listed_tensors(trajectory[idx-1], trajectory[idx+1])
                trajectory[-1] = mean_two_through_listed_tensors(trajectory[-2], trajectory[-1])
                
                waypoint_i_piece_list = [trajectory[idx] for idx in waypoints_idx]
                waypoint_list.append(waypoint_i_piece_list)
            else:
                if args.abl_uniform_points:
                    step_size, waypoints = get_step_and_waypoint(trajectory)
                    # step_size_list.append(step_size)
                    
                    total_len_step_size = len(step_size)-1
                    
                    uni_step_size = []
                    for idx, layer in enumerate(step_size):
                        uni_layer = []
                        for w in layer:
                            # w = idx / total_len_step_size
                            _w = torch.tensor(idx / total_len_step_size, device=w.device)
                            uni_layer.append(_w)
                        uni_step_size.append(uni_layer)
                    
                    step_size_list.append(uni_step_size)
                    waypoint_list.append(waypoints)
                elif args.abl_decreasing_points:
                    step_size, waypoints = get_step_and_waypoint(trajectory)
                    
                    '''
                    step_size = [51, 50, 49, ..., 1]
                    step_size = step_size / sum(step_size)
                    
                    '''
                    
                    _list_sum = sum([i for i in range(1, len(step_size)+1)])
                    
                    desc_step_size = []
                    last_sum = 0
                    for idx, layer in enumerate(step_size):
                        desc_layer = []
                        for w in layer:
                            _w = torch.tensor(((len(step_size) - idx) / _list_sum) + last_sum , device=w.device)
                            desc_layer.append(_w)
                        last_sum += (len(step_size) - idx) / _list_sum
                        desc_step_size.append(desc_layer)
                        
                    step_size_list.append(desc_step_size)
                    waypoint_list.append(waypoints)

                elif args.abl_projected_points:
                    step_size, waypoints = get_step_and_waypoint(trajectory)
                    
                    '''
                    
                    '''
                    p_0 = torch.cat([layer.flatten() for layer in waypoints[0]])
                    p_n = torch.cat([layer.flatten() for layer in waypoints[-1]])
                    V_T_star = p_n - p_0
                    
                    projected_step_size = []
                    for idx, layer in enumerate(trajectory):
                        p_i = torch.cat([l.flatten() for l in layer])
                        V_T_i = p_i - p_0
                        _w = [torch.dot(V_T_i, V_T_star) / torch.dot(V_T_star, V_T_star) for _ in range(len(layer))]
                        projected_step_size.append(_w)
                        
                    step_size_list.append(projected_step_size)
                    waypoint_list.append(waypoints)
                    
                    pass
                else:
                    step_size, waypoints = get_step_and_waypoint(trajectory)
                    step_size_list.append(step_size)
                    waypoint_list.append(waypoints)
        
        # save acc_sum_lay_list for current buffer
        print("Saving buffer {}...".format(file_idx))
        if args.num_interpolation > 2:
            torch.save(waypoint_list, os.path.join(expert_dir, "waypoint_buffer_ip{}_{}.pt".format(args.num_interpolation, file_idx)))
            torch.save(step_size_list, os.path.join(expert_dir, "step_buffer_ip{}_{}.pt".format(args.num_interpolation, file_idx)))
            print(os.path.join(expert_dir, "waypoint_buffer_ip{}_{}.pt".format(args.num_interpolation, file_idx)))
            print(os.path.join(expert_dir, "step_buffer_ip{}_{}.pt".format(args.num_interpolation, file_idx)))
        else:
            if args.abl_uniform_points:
                torch.save(waypoint_list, os.path.join(expert_dir, "waypoint_buffer_uni_{}.pt".format(file_idx)))
                torch.save(step_size_list, os.path.join(expert_dir, "step_buffer_uni_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "waypoint_buffer_uni_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "step_buffer_uni_{}.pt".format(file_idx)))
            elif args.abl_decreasing_points:
                torch.save(waypoint_list, os.path.join(expert_dir, "waypoint_buffer_dec_{}.pt".format(file_idx)))
                torch.save(step_size_list, os.path.join(expert_dir, "step_buffer_dec_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "waypoint_buffer_dec_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "step_buffer_dec_{}.pt".format(file_idx)))
            elif args.abl_projected_points:
                torch.save(waypoint_list, os.path.join(expert_dir, "waypoint_buffer_proj_{}.pt".format(file_idx)))
                torch.save(step_size_list, os.path.join(expert_dir, "step_buffer_proj_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "waypoint_buffer_proj_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "step_buffer_proj_{}.pt".format(file_idx)))
            else:
                torch.save(waypoint_list, os.path.join(expert_dir, "waypoint_buffer_{}.pt".format(file_idx)))
                torch.save(step_size_list, os.path.join(expert_dir, "step_buffer_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "waypoint_buffer_{}.pt".format(file_idx)))
                print(os.path.join(expert_dir, "step_buffer_{}.pt".format(file_idx)))


def interpolate_weight(start_weight, end_weight, alpha):
    """
    interpolate some points between start and end point
    usage e.g.: theta = get_waypoint(traj_info[t_i][0], traj_info[t_i][1], weight_info[t_i][i])
    :param start_weight: list of weights (tensor) for each layer
    :param end_weight: list of weights (tensor) for each layer
    :param alpha: float
    :return:
    """
    return [torch.lerp(start, end, alpha) for start, end in zip(start_weight, end_weight)]


def test_compression(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100", "Tiny"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("dataset: {}, model: {}".format(args.dataset, args.model))
    
    expert_dir = save_dir
    expert_files = []
    n = 1
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    
    
    buffer_idx = 2
    waypoints_list = torch.load(os.path.join(save_dir, f"waypoint_buffer_ip{args.num_interpolation}_{buffer_idx}.pt"))
    step_size_list = torch.load(os.path.join(save_dir, f"step_buffer_ip{args.num_interpolation}_{buffer_idx}.pt"))
    buffer = torch.load(expert_files[buffer_idx])[1]
    
    # compsd_traj = CompressedTrajectory(traj_info[0][0], traj_info[0][1], weight_info[0])
    compsd_traj = CompressedTrajectoryWithInterpolation(waypoints_list[1], step_size_list[1], num_interpolation=args.num_interpolation)
    
    
    from exps.visualize_synthetic import eval_theta
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    acc_conv_list = []
    acc_orig_list = []
    start_ep = 35
    # for i in range(start_ep, start_ep+13):
    try:
        for i in range(compsd_traj.waypoints_idx[-1] + 1):
            theta = compsd_traj.get_point_n(i)
            theta = [layer.to(args.device) for layer in theta]
            
            tool_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            from reparam_module import ReparamModule
            tool_net = ReparamModule(tool_net).eval()
            l, a = eval_theta(theta, testloader, tool_net, args)
            
            # print("i={}, \tloss={}, \tacc={}".format(i, round(l, 3), round(a, 4)))
            acc_conv_list.append(a)
            
            theta = buffer[i]
            theta = torch.cat([p.reshape(-1) for p in theta], 0)
            l, a = eval_theta(theta, testloader, tool_net, args)
            acc_orig_list.append(a)
            
            print("i={}, \tacc_orig={}, \tacc_conv={}".format(i, round(acc_orig_list[-1], 4), round(acc_conv_list[-1], 4)))
    except Exception as e:
        print(e)
    
    # import matplotlib.pyplot as plt
    # # title: Convexified trajectory
    # # plt.title("Acc-Epoch")
    # # set to 橙色
    # # plt.plot(acc_orig_list, label="Original", color='orange')
    # # plt.plot(acc_conv_list, label="Convexified", color='blue')
    # plt.figure(dpi=300)
    # plt.plot(acc_orig_list, label="Original", color='#00b050', marker='o', markerfacecolor='#ed7d31',
    #          markersize=12, markeredgecolor='#72401f')
    # plt.plot(acc_conv_list, label="Convexified", color='#00aff0', marker='^', markerfacecolor='#4472c4',
    #          markersize=12, markeredgecolor='#27395c')
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend()
    # plt.show()
    import matplotlib.pyplot as plt
    # title: Convexified trajectory
    # set to 橙色
    # plt.plot(acc_orig_list, label="Original", color='orange')
    # plt.plot(acc_conv_list, label="Convexified", color='blue')
    plt.figure(dpi=400)

    plt.plot(acc_orig_list, label="Original", color='#00b050', marker='o', markerfacecolor='#ed7d31',
             markersize=6, markeredgecolor='#72401f', linestyle='dashed')
    plt.plot(acc_conv_list, label="Convexified", color='#00aff0', marker='^', markerfacecolor='#4472c4',
             markersize=6, markeredgecolor='#27395c')
    plt.title("Acc-Epoch, ip={}".format(args.num_interpolation))
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()
    
    print("Done!")
    pass


def acc_curve_compare():
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100", "Tiny"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("dataset: {}, model: {}".format(args.dataset, args.model))
    
    ori_trajectory = torch.load(os.path.join(save_dir, "replay_buffer_0.pt"))[1]
    
    # MCT_trajectory
    MCT_waypoints_list = torch.load(os.path.join(save_dir, f"waypoint_buffer_0.pt"))
    MCT_step_size_list = torch.load(os.path.join(save_dir, f"step_buffer_0.pt"))
    MCT_compressed_traj = CompressedTrajectoryWithInterpolation(MCT_waypoints_list[1], MCT_step_size_list[1], num_interpolation=2)
    
    # uniform_trajectory
    uni_waypoints_list = torch.load(os.path.join(save_dir, f"waypoint_buffer_uni_0.pt"))
    uni_step_size_list = torch.load(os.path.join(save_dir, f"step_buffer_uni_0.pt"))
    uni_compressed_traj = CompressedTrajectoryWithInterpolation(uni_waypoints_list[1], uni_step_size_list[1], num_interpolation=2)
    
    # decreasing_trajectory
    dec_waypoints_list = torch.load(os.path.join(save_dir, f"waypoint_buffer_dec_0.pt"))
    dec_step_size_list = torch.load(os.path.join(save_dir, f"step_buffer_dec_0.pt"))
    dec_compressed_traj = CompressedTrajectoryWithInterpolation(dec_waypoints_list[1], dec_step_size_list[1], num_interpolation=2)
    
    # projected_trajectory
    proj_waypoints_list = torch.load(os.path.join(save_dir, f"waypoint_buffer_proj_0.pt"))
    proj_step_size_list = torch.load(os.path.join(save_dir, f"step_buffer_proj_0.pt"))
    proj_compressed_traj = CompressedTrajectoryWithInterpolation(proj_waypoints_list[1], proj_step_size_list[1], num_interpolation=2)
    
    from exps.visualize_synthetic import eval_theta
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    
    acc_ori_list = []
    acc_MCT_list = []
    acc_uni_list = []
    acc_dec_list = []
    acc_proj_list = []
    
    try:
        for i in range(len(ori_trajectory)):
            tool_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            from reparam_module import ReparamModule
            tool_net = ReparamModule(tool_net).eval()
            
            theta = ori_trajectory[i]
            theta = torch.cat([p.reshape(-1) for p in theta], 0)
            
            l, a = eval_theta(theta, testloader, tool_net, args)
            acc_ori_list.append(a)
            
            theta = MCT_compressed_traj.get_point_n(i)
            theta = [layer.to(args.device) for layer in theta]
            
            l, a = eval_theta(theta, testloader, tool_net, args)
            acc_MCT_list.append(a)
            
            theta = uni_compressed_traj.get_point_n(i)
            theta = [layer.to(args.device) for layer in theta]
            
            l, a = eval_theta(theta, testloader, tool_net, args)
            acc_uni_list.append(a)
            
            theta = dec_compressed_traj.get_point_n(i)
            theta = [layer.to(args.device) for layer in theta]
            
            l, a = eval_theta(theta, testloader, tool_net, args)
            acc_dec_list.append(a)
            
            theta = proj_compressed_traj.get_point_n(i)
            theta = [layer.to(args.device) for layer in theta]
            
            l, a = eval_theta(theta, testloader, tool_net, args)
            
            acc_proj_list.append(a)
            
            print("i={}, \tacc_ori={}, \tacc_MCT={}, \tacc_uni={}, \tacc_dec={}, \tacc_proj={}"
                  .format(i, round(acc_ori_list[-1], 4),
                          round(acc_MCT_list[-1], 4),
                          round(acc_uni_list[-1], 4),
                          round(acc_dec_list[-1], 4),
                          round(acc_proj_list[-1], 4))
                  )
    except Exception as e:
        print(e)
        

    import matplotlib.pyplot as plt
    
    MAX_LEN = min(len(acc_ori_list), len(acc_MCT_list), len(acc_uni_list), len(acc_dec_list), len(acc_proj_list))
    plt.figure(dpi=400)
    plt.plot(acc_ori_list[:MAX_LEN], label="Original", color='#00b050',
                markersize=6, markeredgecolor='#72401f')
    plt.plot(acc_MCT_list[:MAX_LEN], label="MCT", color='#00aff0',
                markersize=6, markeredgecolor='#27395c')
    plt.plot(acc_uni_list[:MAX_LEN], label="Uniform", color='#ff0000',
                markersize=6, markeredgecolor='#ff0000')
    plt.plot(acc_dec_list[:MAX_LEN], label="Decreasing", color='#ff00ff',
                markersize=6, markeredgecolor='#ff00ff')
    plt.plot(acc_proj_list[:MAX_LEN], label="Projected", color='#0000ff',
                markersize=6, markeredgecolor='#0000ff')
    plt.title("Acc-Epoch")
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for Convexify')
    
    # buffer path
    # parser.add_argument('--buffer', type=str, default="../buffers/CIFAR10_NO_ZCA/ConvNet/",
    #                     help='Path to the buffer file')

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
    # parser.add_argument('--save_interval', type=int, default=10)
    
    # alpha
    # parser.add_argument("--alpha", type=float, default=0.3, help="alpha for interpolation")
    
    parser.add_argument('--num_interpolation', type=int, default=4, help='number of interpolation points')
    
    parser.add_argument('--abl_uniform_points', action='store_true', help='whether to use uniform points')
    parser.add_argument('--abl_decreasing_points', action='store_true', help='whether to use decreasing points')
    parser.add_argument('--abl_projected_points', action='store_true', help='whether to use projected points')
    
    
    args = parser.parse_args()
    
    main(args)
    # test_compression(args)
    
    # acc_curve_compare()


'''
CF10
--num_interpolation=4  --zca  --dataset=CIFAR10

CF100
--num_interpolation=4  --zca  --dataset=CIFAR100


TinyImageNet
--dataset=Tiny --data_path=../data/tiny-imagenet-200
'''


'''
Ablation of \Beta

## uniform points
nohup python -u distill_compressed.py --dataset=CIFAR10 --abl_uniform_points --num_interpolation=2 --zca --ipc=50 --syn_steps=30 --expert_epochs=5 --max_start_epoch=30 --lr_img=1e3 --lr_lr=1e-05 --lr_init=1e-3 > abl_beta_uni_distill_cifar10_ipc50.log 2>&1 &

## original
nohup python -u distill_compressed.py --dataset=CIFAR10 --num_interpolation=2 --zca --ipc=50 --syn_steps=30 --expert_epochs=5 --max_start_epoch=30 --lr_img=1e3 --lr_lr=1e-05 --lr_init=1e-3 > abl_beta_ori_distill_cifar10_ipc50.log 2>&1 &


'''