# load the Torch library and other utilities
#----------------------------------------------------

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision import models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#import pandas
import time
import os
import pdb
import sys
import scipy.io
from config import get_cfg_defaults
from utils.normalization_utils import get_imagenet_mean_std_normalized
from networks import alexnet, vggnet, resnet, inceptionv1


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

def parse_args():
  
    parser = argparse.ArgumentParser(description='classification network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg  = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()    
    # print(cfg)
    
    # create the directory where the trained model will be saved!
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        print(f"Directory '{cfg.OUTPUT_DIR}' created.")
    else:
        print(f"Directory '{cfg.OUTPUT_DIR}' already exists.")    
    
    return args, cfg


def prepare_data(config):
    
    mean, std = get_imagenet_mean_std_normalized()
    print(f"mean: {mean}, std: {std}")
    
    # For fine-tuning with an AlexNet/VGG/ResNet architecture that has been pre-trained using the ImageNet dataset, 
    # you need to normalize each image with the given mean and standard deviation.
    transform = transforms.Compose([
        transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) # ImageNet: mean (R, G, B) and standard deviation (R, G, B)
    ])

    train_dir       = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET) #'/content/drive/MyDrive/cs195_fall24/datasets/bcdp_v1/train'
    test_dir        = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET) #'/content/drive/MyDrive/cs195_fall24/datasets/bcdp_v1/test'

    # it loads images from the given directory, subsequently, classes are assigned labels according to the sorted order of the folder names.
    train_dataset   = datasets.ImageFolder(train_dir, transform=transform) 
    # it loads images from the given directory, subsequently, classes are assigned labels according to the sorted order of the folder names.
    test_dataset    = datasets.ImageFolder(test_dir,  transform=transform) 

    N_train         = len(train_dataset)
    N_test          = len(test_dataset)

    print("Size of train set:", N_train)
    print("Size of test set:",  N_test)
    
    # shuffle the images in training set during fine-tuning
    train_dataloader  = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True) 
    # you don't need to shuffle test images as they are not used during training
    test_dataloader   = DataLoader(test_dataset, batch_size=config.TEST.BATCH_SIZE,  shuffle=False) 
    

    return train_dataloader, test_dataloader

    


def main():

    # get the arguments after parsing
    args, config = parse_args()
    if args.seed > 0:
        print('seeding with ', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    
    # prepare data
    train_dataloader, test_dataloader = prepare_data(config)
    
    
    # You should change this to the appropriate number when you fine-tune your model on a different dataset.
    number_of_classes = config.DATASET.NUM_CLASSES
    
    if config.MODEL.NAME == 'alexnet':
        model = alexnet.AlexNet(number_of_classes)
    elif config.MODEL.NAME == 'vggnet':
        model = vggnet.VGGNet(number_of_classes)
    elif config.MODEL.NAME == 'resnet':
        model = resnet.ResNet152(number_of_classes)
    elif config.MODEL.NAME == 'inceptionv1':
        model = inceptionv1.InceptionV1(number_of_classes)
    model.to(device)
    print(model)
    
    
    print(f"{config.MODEL.NAME} model has been fine-tuned!")
    print("Total fine-tuning time: %.3f sec" %( (time.time()-start_time)) )
    print("Total fine-tuning time: %.3f hrs" %( (time.time()-start_time)/3600) )
    
    # save the losses and other meta information
    data = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'best_accuracy': best_accuracy,
    'best_epoch': best_epoch
    }
    scipy.io.savemat(os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '_meta_data.mat'), 
                    data)
    
    
main()
