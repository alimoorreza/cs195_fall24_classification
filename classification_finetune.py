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
from config import get_cfg_defaults
from utils.normalization_utils import get_imagenet_mean_std_normalized
from networks import alexnet


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

    
def train_loop(dataloader, model, loss_fn, optimizer):

    # for calculating the probability of the network prediction.
    softmax         = nn.Softmax(dim=1)

    size            = len(dataloader.dataset)
    num_batches     = len(dataloader)

    model.train()                   # set the model to training mode for best practices

    train_loss      = 0
    correct         = 0
    train_pred_all  = []
    train_y_all     = []

    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss

        # ----------- putting data into gpu or sticking to cpu ----------
        X = X.to(device)     # send data to the GPU device (if available)
        y = y.to(device)
        # -----------                                         ----------

        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # compute the accuracy
        pred_prob   = softmax(pred)
        pred_y 			= torch.max(pred_prob, 1)[1]
        train_correct = (pred_y == y).sum()
        correct    += train_correct.data

        train_pred_all.append(pred_y) # save predicted output for the current batch
        train_y_all.append(y)         # save ground truth for the current batch

    train_pred_all = torch.cat(train_pred_all) # need to concatenate batch-wise appended items
    train_y_all = torch.cat(train_y_all)

    train_loss = train_loss/num_batches
    correct    = correct.cpu().numpy()/size

    print('Confusion matrix for training set:\n', confusion_matrix(train_y_all.cpu().data, train_pred_all.cpu().data))
    return train_loss, 100*correct


def test_loop(dataloader, model, loss_fn):

    # for calculating the probability of the network prediction.
    softmax         = nn.Softmax(dim=1)
    
    model.eval()                    # set the model to evaluation mode for best practices

    size                = len(dataloader.dataset)
    num_batches         = len(dataloader)
    test_loss, correct  = 0, 0
    test_pred_all       = []
    test_y_all          = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():

        for X, y in dataloader:

            # ----------- putting data into gpu or sticking to cpu ----------
            X = X.to(device)     # send data to the GPU device (if available)
            y = y.to(device)
            # -----------                                         ----------

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # calculate probability and save the outputs for confusion matrix computation
            pred_prob     = softmax(pred)
            pred_y        = torch.max(pred_prob, 1)[1]
            test_correct  = (pred_y == y).sum()
            correct      += test_correct.data

            test_pred_all.append(pred_y) # save predicted output for the current batch
            test_y_all.append(y)         # save ground truth for the current batch


    #pdb.set_trace()
    test_pred_all = torch.cat(test_pred_all)
    test_y_all = torch.cat(test_y_all)

    test_loss = test_loss/num_batches
    correct   = correct.cpu().numpy()/size
    print(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print('Confusion matrix for test set:\n', confusion_matrix(test_y_all.cpu().data, test_pred_all.cpu().data))
    return test_loss, 100*correct, confusion_matrix(test_y_all.cpu().data, test_pred_all.cpu().data)


def main():

    # get the arguments after parsing
    args, config = parse_args()
    if args.seed > 0:
        print('seeding with ', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    
    # prepare data
    train_dataloader, test_dataloader = prepare_data(config)
    
    
    # load hyper-parameters
    loss_fn           = nn.CrossEntropyLoss()     # criterion    
    
    
    # Headsup! You should change this to the appropriate number when you fine-tune your model on a different dataset.
    number_of_classes = config.DATASET.NUM_CLASSES
    
    if config.MODEL.NAME == 'alexnet':
        model = alexnet.AlexNet(number_of_classes)
    elif config.MODEL.NAME == 'vgg':
        model = VGGNet(number_of_classes)
    model.to(device)
    print(model)
    
    
    # cudnn related setting
    '''
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    '''
    gpus  = list(config.GPUS)
    '''
    model = FullModel(model, criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    logger.info(f'Using DataParallel')
    '''

    
    
    # optimizer
    if config.TRAIN.OPTIMIZER == 'adam':
        
        optimizer         = torch.optim.Adam(model.parameters(), 
                                             lr=config.TRAIN.BASE_LR)
    elif config.TRAIN.OPTIMIZER == 'sgd':
        
        optimizer         = torch.optim.SGD(model.parameters(),
                                    lr=config.TRAIN.BASE_LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WEIGHT_DECAY,
                                    nesterov=config.TRAIN.NESTEROV)
    else:
        raise ValueError('Only Support ADAM and SGD optimizer')
        
        
    epochs              = config.TRAIN.END_EPOCH        
    train_losses        = []
    test_losses         = []
    train_accuracies    = []
    test_accuracies     = []
    start_time          = time.time()
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        avg_train_loss, train_accuracy                    = train_loop(train_dataloader, model, loss_fn, optimizer)
        avg_test_loss, test_accuracy, conf_matrix_test    = test_loop(test_dataloader,   model, loss_fn)
        # save the losses and accuracies
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    print(f"{config.MODEL.NAME} model has been fine-tuned!")
    print("Total fine-tuning time: %.3f sec" %( (time.time()-start_time)) )
    print("Total fine-tuning time: %.3f hrs" %( (time.time()-start_time)/3600) )

    '''
    # visualizing the loss curves
    plt.plot(range(1,epochs+1), train_losses)
    plt.plot(range(1,epochs+1), test_losses)
    plt.title('AlexNet average losses after each epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    '''
        


main()
