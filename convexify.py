import argparse
import os

import math
import torch


def minus_through_listed_tensors(A, B):
    return [a - b for a, b in zip(A, B)]

def mean_two_through_listed_tensors(A, B):
    return [torch.div(torch.add(a, b), 2) for a, b in zip(A, B)]


def check_if_nan(_trajectory):
    for i in range(len(_trajectory)):
        for j in range(len(_trajectory[i])):
            if torch.isnan(_trajectory[i][j]).any():
                print("NaN detected at trajectory[{}][{}]".format(i, j))
                return True
    return False


def convexify_a_trajectory(_trajectory):
    # todo: adapt to torch.lerp
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
def convexify_trajectory_with_interpolation(_trajectory, idxs=(6, 25)):
    # _trajectory = copy.deepcopy(_trajectory)
    _target_trajectory = []
    
    # smooth the waypoints by their neighbors
    for i in range(len(idxs)):
        tmp = mean_two_through_listed_tensors(_trajectory[idxs[i] + 1], _trajectory[idxs[i] - 1])
        _trajectory[idxs[i]] = mean_two_through_listed_tensors(_trajectory[idxs[i]], tmp)
    
    # smooth the final point
    tmp = mean_two_through_listed_tensors(_trajectory[-3], _trajectory[-2])
    _trajectory[-1] = mean_two_through_listed_tensors(_trajectory[-1], tmp)
    
    _target_trajectory.extend(convexify_a_trajectory(_trajectory[:idxs[0] + 1]))
    for i in range(len(idxs) - 1):
        _target_trajectory.extend(convexify_a_trajectory(_trajectory[idxs[i]:idxs[i + 1] + 1])[1:])
    _target_trajectory.extend(convexify_a_trajectory(_trajectory[idxs[-1]:])[1:])
    
    return _target_trajectory

def convexify(args):
    """ load the buffer """
    expert_dir = args.buffer
    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    for file_idx, file in enumerate(expert_files):
        print("Processing Buffer {} in {}".format(file_idx, expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        
        print("{} trajectories loaded".format(len(buffer)))
        
        if args.mode == 'normal':
            ''' convexify the buffer '''
            convexified_buffer = []
            for trajectory in buffer:
                # todo: may mean the final_theta
                #  i.e.
                #  trajectory[-1] = mean(trajectory[-1], trajectory[-2])
                
                target_trajectory = convexify_a_trajectory(trajectory)
                
                if check_if_nan(target_trajectory):
                    print("NaN detected in the convexified buffer")
                    exit()
                
                # difference of the final theta
                print('Difference threshold of the final theta: ',
                      max([max((tar-tra).flatten().tolist()) for tar, tra in zip(target_trajectory[-1], trajectory[-1])]))
                convexified_buffer.append(target_trajectory)
                
            ''' save the convexified buffer '''
            print("Saving the convexified buffer")
            torch.save(convexified_buffer, os.path.join(expert_dir, "convexified_replay_buffer_{}.pt".format(file_idx)))
        elif args.mode == 'interpolation':
            ''' convexify the buffer '''
            convexified_buffer = []
            for trajectory in buffer:
                target_trajectory = convexify_trajectory_with_interpolation(trajectory)
                
                if check_if_nan(target_trajectory):
                    print("NaN detected in the convexified buffer")
                    exit()
                
                # difference of the final theta
                print('Difference threshold of the final theta: ',
                      max([max((tar-tra).flatten().tolist()) for tar, tra in zip(target_trajectory[-1], trajectory[-1])]))
                convexified_buffer.append(target_trajectory)
                
            ''' save the convexified buffer '''
            print("Saving the convexified buffer v2 as convexified_v2_replay_buffer_{}.pt".format(file_idx))
            torch.save(convexified_buffer, os.path.join(expert_dir, "convexified_v2_replay_buffer_{}.pt".format(file_idx)))
        elif args.mode == 'v3':
            ''' convexify the buffer '''
            convexified_buffer = []
            for trajectory in buffer:
                target_trajectory = convexify_trajectory_with_interpolation(trajectory, idxs=(15, 30))
                
                if check_if_nan(target_trajectory):
                    print("NaN detected in the convexified buffer")
                    exit()
                
                # difference of the final theta
                print('Difference threshold of the final theta: ',
                      max([max((tar-tra).flatten().tolist()) for tar, tra in zip(target_trajectory[-1], trajectory[-1])]))
                convexified_buffer.append(target_trajectory)
                
            ''' save the convexified buffer '''
            print("Saving the convexified buffer v3 as convexified_v3_replay_buffer_{}.pt".format(file_idx))
            torch.save(convexified_buffer, os.path.join(expert_dir, "convexified_v3_replay_buffer_{}.pt".format(file_idx)))
    
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for Convexify')
    
    # buffer path
    parser.add_argument('--buffer', type=str, default="./buffers/CIFAR10_NO_ZCA/ConvNet/", help='Path to the buffer file')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to load')
    parser.add_argument('--max_experts', type=int, help='Maximum number of experts to load')
    # mode=interpolation/normal
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'interpolation', 'v3'],
                        help='Convexify mode')
    
    args = parser.parse_args()
    convexify(args)
    
    print("exit...")
    
    
"""
run command:
python -u convexify.py --mode interpolation
"""