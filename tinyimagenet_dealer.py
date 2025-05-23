import glob
import os
from shutil import move
from os import rmdir

if __name__ == '__main__':

    target_folder = './data/tiny-imagenet-200/val/'
    
    val_dict = {}
    with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]
    
    paths = glob.glob('./data/tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')
    
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)
    
    rmdir('./data/tiny-imagenet-200/val/images')