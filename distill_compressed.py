import os
import argparse

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
from exps.trajectory_compression import CompressedTrajectoryWithInterpolation

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
               project="Distill_{}_{}".format(args.dataset, args.ipc) if not args.ablation else "Ablation",
               job_type="CleanRepo",
               config=args,
               # name="TESLA_CompressedTraj_ip{}".format(args.num_interpolation),
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    if args.optimizer == 'Adam':
        optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    elif args.optimizer == 'SGD':
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    else:
        raise ValueError('optimizer not supported')
    
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()
    
    scheduler_img = torch.optim.lr_scheduler.StepLR(optimizer_img, step_size=2000, gamma=0.5)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100", "Tiny"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    # file_prefix = "convexified_replay_buffer"
        
    if args.load_all:
        exit()
        # todo: haven't implemented this yet
        # buffer = []
        # n = 0
        # while os.path.exists(os.path.join(expert_dir, "{}_{}.pt".format(file_prefix, n))):
        #     buffer = buffer + torch.load(os.path.join(expert_dir, "{}_{}.pt".format(file_prefix, n)))
        #     n += 1
        # if n == 0:
        #     raise AssertionError("No buffers detected at {}".format(expert_dir))
    else:
        waypoint_files = []
        step_files = []
        assert args.num_interpolation >= 2
        if args.num_interpolation == 2:
            if args.abl_uniform_points:
                # waypoint_buffer_uni_{idx}.pt
                waypoint_file_prefix = "waypoint_buffer_uni"
                step_file_prefix = "step_buffer_uni"
            elif args.abl_decreasing_points:
                waypoint_file_prefix = "waypoint_buffer_dec"
                step_file_prefix = "step_buffer_dec"
            elif args.abl_projected_points:
                waypoint_file_prefix = "waypoint_buffer_proj"
                step_file_prefix = "step_buffer_proj"
                
            else:
                waypoint_file_prefix = "waypoint_buffer"
                step_file_prefix = "step_buffer"
        else:
            waypoint_file_prefix = "waypoint_buffer_ip{}".format(args.num_interpolation)
            step_file_prefix = "step_buffer_ip{}".format(args.num_interpolation)
        n = 0
        while os.path.exists(os.path.join(expert_dir, "{}_{}.pt".format(waypoint_file_prefix,n))) and \
            os.path.exists(os.path.join(expert_dir, "{}_{}.pt".format(step_file_prefix, n))):
            
            waypoint_files.append(os.path.join(expert_dir, "{}_{}.pt".format(waypoint_file_prefix,n)))
            step_files.append(os.path.join(expert_dir, "{}_{}.pt".format(step_file_prefix, n)))
            n += 1
        
        buffer_files = [[waypoint_files[i], step_files[i]] for i in range(len(waypoint_files))]
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        
        random.shuffle(buffer_files)
        
        if args.max_files is not None:
            buffer_files = buffer_files[:args.max_files]
        print("loading file {}".format(buffer_files[file_idx][0]))
        print("loading file {}".format(buffer_files[file_idx][1]))
        waypoint_buffer = torch.load(buffer_files[file_idx][0])
        step_buffer = torch.load(buffer_files[file_idx][1])
        
        buffer_both = [[waypoint_buffer[i], step_buffer[i]] for i in range(len(waypoint_buffer))]
        
        if args.max_experts is not None:
            buffer_both = buffer_both[:args.max_experts]
        random.shuffle(buffer_both)
        
    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}
    
    LowerBound = 0
    
    for it in range(0, args.Iteration+1):
        save_this_it = False
        
        if not args.mute_fub:
            if args.ipc >= 50:
                UpperBound = (it / (args.Iteration - 2500)) * args.max_start_epoch + 20
                # LowerBound = (it / (args.Iteration - 1000)) * 10
                # LowerBound = min(10, LowerBound)
            elif args.ipc >= 10 and args.dataset in ("CIFAR100", "Tiny"):
                UpperBound = (it / (args.Iteration - 2500)) * args.max_start_epoch + 15
            else:
                UpperBound = (it / (args.Iteration - 2500)) * args.max_start_epoch + 0.5
            UpperBound = min(UpperBound, args.max_start_epoch, 50-args.expert_epochs)
        else:
            UpperBound = args.max_start_epoch
        
        if not args.mute_continuous_spl:
            start_epoch = np.random.uniform(LowerBound, UpperBound)
        else:
            start_epoch = np.random.randint(LowerBound, UpperBound)
        target_epoch = start_epoch + args.expert_epochs
        target_epoch = min(target_epoch, 50)
        
        wandb.log({"UpperBound": UpperBound}, step=it)
        # if args.ipc >= 50:
        #     wandb.log({"LowerBound": LowerBound}, step=it)

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, "convexified_v1")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)


                torch.save(image_save.cpu(), os.path.join(save_dir, "images_ipc-{}_{}.pt".format(args.ipc, it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_ipc-{}_{}.pt".format(args.ipc, it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_ipc-{}_best.pt".format(args.ipc)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_ipc-{}_best.pt".format(args.ipc)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        if False:
                            torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            # todo: haven't implemented this yet
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory_waypoints, expert_step = buffer_both[expert_idx]
            expert_idx += 1
            # if expert_idx == len(waypoint_buffer):
            if expert_idx == len(buffer_both):
                expert_idx = 0
                if args.max_files != 1:
                    file_idx += 1
                # if file_idx == len(waypoint_files):
                if file_idx == len(buffer_files):
                    file_idx = 0
                    random.shuffle(buffer_files)
                print("loading file {}".format(buffer_files[file_idx][0]))
                print("loading file {}".format(buffer_files[file_idx][1]))
                if args.max_files != 1:
                    del waypoint_buffer
                    del step_buffer
                    del buffer_both
                    waypoint_buffer = torch.load(buffer_files[file_idx][0])
                    step_buffer = torch.load(buffer_files[file_idx][1])
                    buffer_both = [[waypoint_buffer[i], step_buffer[i]] for i in range(len(waypoint_buffer))]
                if args.max_experts is not None:
                    buffer_both = buffer_both[:args.max_experts]
                random.shuffle(buffer_both)
        
        compsdTraj = CompressedTrajectoryWithInterpolation(expert_trajectory_waypoints, expert_step, num_interpolation=args.num_interpolation)
        
        starting_params = compsdTraj.get_point_n(start_epoch)
        
        if args.noise_start and args.noise_start > 0.:
            # add small noise to the starting parameters
            for p in starting_params:
                # if p is 1. or zero, don't add noise
                if torch.sum(torch.abs(p)) == 0.:
                    continue
                if torch.sum(torch.abs(p)) == 1.:
                    continue
                
                # set the epsilon to be 5% of the parameter value
                epsilons = torch.abs(p) * args.noise_start
                noise = torch.randn_like(p) * epsilons
                p += noise

        target_params = compsdTraj.get_point_n(target_epoch)
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        
        param_dist = torch.tensor(0.0).to(args.device)
        
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        
        # note: fixed a bug.
        # todo: may use a better model to generate soft labels.
        # produce soft labels for soft label assignment.
        if args.teacher_label:
            label_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(
                args.device)  # get a random model
            label_net = ReparamModule(label_net)
            label_net.eval()
            
            # use the target param as the model param to get soft labels.
            # label_params = copy.deepcopy(target_params.detach()).requires_grad_(False)
            label_params = compsdTraj.get_point_n(args.max_start_epoch)
            label_params = torch.cat([p.data.to(args.device).reshape(-1) for p in label_params], 0).requires_grad_(False)
            
            batch_labels = []
            SOFT_INIT_BATCH_SIZE = 50
            if image_syn.shape[0] > SOFT_INIT_BATCH_SIZE and args.dataset == 'ImageNet':
                for indices in torch.split(torch.tensor([i for i in range(0, image_syn.shape[0])], dtype=torch.long),
                                           SOFT_INIT_BATCH_SIZE):
                    batch_labels.append(label_net(image_syn[indices].detach().to(args.device), flat_param=label_params))
                label_syn = torch.cat(batch_labels, dim=0)
            else:
                label_syn = label_net(image_syn.detach().to(args.device), flat_param=label_params)
            # label_syn = torch.cat(batch_labels, dim=0)
            label_syn = torch.nn.functional.softmax(label_syn, dim=1)
            
            del label_net, label_params
            for _ in batch_labels:
                del _
        
        syn_images = image_syn
        
        y_hat = label_syn.to(args.device)
        
        syn_image_gradients = torch.zeros(syn_images.shape).to(args.device)
        x_list = []
        original_x_list = []
        y_list = []
        indices_chunks = []
        gradient_sum = torch.zeros(student_params[-1].shape).to(args.device)
        indices_chunks_copy = []
        for _ in range(args.syn_steps):
            
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            
            these_indices = indices_chunks.pop()
            indices_chunks_copy.append(these_indices)
            
            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            original_x_list.append(x)
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            x_list.append(x.clone())
            y_list.append(this_y.clone())
            
            forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)
            
            grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]
            
            detached_grad = grad.detach().clone()
            student_params.append(student_params[-1] - syn_lr.item() * detached_grad)
            gradient_sum += detached_grad
            
            del grad
            
        # torch.cuda.empty_cache()
        
        # --------Compute the gradients regarding input image and learning rate---------
        # compute gradients invoving 2 gradients
        for i in range(args.syn_steps):
            # compute gradients for w_i
            w_i = student_params[i]
            output_i = student_net(x_list[i], flat_param=w_i)
            if args.batch_syn:
                ce_loss_i = criterion(output_i, y_list[i])
            else:
                ce_loss_i = criterion(output_i, y_hat)
            
            grad_i = torch.autograd.grad(ce_loss_i, w_i, create_graph=True, retain_graph=True)[0]
            single_term = syn_lr.item() * (target_params - starting_params)
            square_term = (syn_lr.item() ** 2) * gradient_sum
            gradients = 2 * torch.autograd.grad((single_term + square_term) @ grad_i / param_dist, original_x_list[i])
            with torch.no_grad():
                syn_image_gradients[indices_chunks_copy[i]] += gradients[0]
        # ---------end of computing input image gradients and learning rates--------------
        
        syn_images.grad = syn_image_gradients
        
        grand_loss = starting_params - syn_lr * gradient_sum - target_params
        grand_loss = grand_loss.dot(grand_loss) / param_dist
        
        lr_grad = torch.autograd.grad(grand_loss, syn_lr)[0]
        syn_lr.grad = lr_grad

        optimizer_img.step()
        if not args.mute_syn_lr:
            optimizer_lr.step()
        
        if args.lr_decay:
            scheduler_img.step()
            # scheduler_lr.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
            

    wandb.finish()
    print('Training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    # parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=float, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=float, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--teacher_label', action='store_true', default=False, help='whether to use label from the expert model to guide the distillation process.')

    parser.add_argument('--noise_start', type=float, default=0., help='add small noise to the starting parameters')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--mute_syn_lr', action='store_true', help='mute syn_lr')
    
    # mute floatting upper bound
    parser.add_argument('--mute_fub', action='store_true', default=True, help='float upper bound')
    
    parser.add_argument('--num_interpolation', type=int, default=4, help='number of interpolation points')
    parser.add_argument('--mute_continuous_spl', action='store_true', help='mute continuous sampling')
    
    parser.add_argument('--ablation', action='store_true', help='is ablation study')
    
    # parser.add_argument('--abl_uniform_points', action='store_true', help='uniform interpolation points')
    parser.add_argument('--abl_uniform_points', action='store_true', help='whether to use uniform points')
    parser.add_argument('--abl_decreasing_points', action='store_true', help='whether to use decreasing points')
    parser.add_argument('--abl_projected_points', action='store_true', help='whether to use projected points')


    args = parser.parse_args()

    main(args)


"""
Best performing Hyper-parameters:

Cifar10
ipc | zca | expert_epochs | syn_steps | max_start_epoch | interpolated_point | acc_max | std_max
1   | T   | 5             | 50        | 3               | [0, 6, 25, 50]     | 48.48   | 0.2262
10  | T   | 5             | 30        | 4               | [0, 6, 25, 50]     |    |

"""
