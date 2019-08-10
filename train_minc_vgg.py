import numpy as np
import argparse
import os, time, sys
import logging

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.minc2500 import MINC2500



parser = argparse.ArgumentParser()
parser.add_argument('--train_folder',  type=str,     default='./data/minc-2500/', help="folder path for training dataset")
parser.add_argument('--workers',       type=int,     default=0,              help="thread number for read images")
parser.add_argument('--model',         type=str,     default='vgg19',        help="network model")
parser.add_argument('--class_num',     type=int,     default=23,             help="class_num for training and testing")
parser.add_argument('--batchsize',     type=int,     default=16,             help="batchsize for training and testing")
parser.add_argument('--learning_rate', type=float,   default=1e-3,           help="learning rate for training")
parser.add_argument('--epochs',        type=int,     default=1000,           help="learning epochs for training")
parser.add_argument('--modeldir',      type=str,     default="./model/",     help="model directory")
parser.add_argument('--logdir',        type=str,     default="./log/",       help="log directory")
args = parser.parse_args()



if __name__ == "__main__":

    # Configure logger
    os.makedirs(args.logdir, exist_ok=True)
    logging.getLogger('VGG').setLevel(logging.CRITICAL)
    logging.basicConfig(filename=args.logdir +'vgg.log', filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    root_logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(stdout_handler)
    logging.info("Start program")
    logging.info(torch.__version__)


    #### get datasets
    train_trans = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
    ])

    train_set    =  MINC2500(root_dir=args.train_folder, set_type='train', split=1, transform=train_trans)
    val_set      =  MINC2500(root_dir=args.train_folder, set_type='validate', split=1, transform=val_trans)    
    train_loader =  DataLoader(dataset=train_set, batch_size=args.batchsize, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   =  DataLoader(dataset=val_set,   batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    epoch_iters = len(train_loader)


    #### define model and loss
    net = models.vgg19()
    # net.load_state_dict(torch.load("./model/pretrained/vgg19.pth"))
    net.classifier._modules['6'] = nn.Linear(4096, args.class_num)
    net.load_state_dict(torch.load("./model/vgg_iter10.pth"))
    net.cuda()
    for parma in net.parameters():
        parma.requires_grad = True

    logging.info(net)
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))            
    optimizer = torch.optim.SGD(net.parameters(), lr = args.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss().cuda()                      


    #### training 
    dtype = torch.cuda.FloatTensor
    for epoch in range(0, args.epochs):

        ## evaluation
        net.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(val_loader):
            images        = Variable(images, requires_grad=False).type(dtype)
            outputs       = net(images)
            _, predicted  = torch.max(outputs.data, 1)
            correct      += (predicted.cpu() == labels.cpu()).sum().numpy().astype('float')
            total        += labels.cpu().size()[0]
            # if i > 100:  ## 取100×16个valiadation
            #     break
        acc = correct / total
        logging.info("[*] %d / %d, Validation acc: %.3f" % ((epoch+1), args.epochs,  acc))

        ## Train
        start_epoch = time.time()
        net.train()
        correct = 0
        total = 0
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images       = Variable(images, requires_grad=False).type(dtype)
            labels       = Variable(labels, requires_grad=False).cuda()

            optimizer.zero_grad()
            outputs      = net(images)
            loss         = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            cur_correct  = (predicted.cpu() == labels.cpu()).sum().numpy().astype('float')
            cur_total    = labels.cpu().size()[0]
            cur_acc      = cur_correct / cur_total
            correct     += cur_correct
            total       += cur_total
            total_loss  += loss.item() / epoch_iters

            if i % 50 == 0:
                logging.info("%d epoch, %d / %d iter, loss: %.5f, acc: %.3f " % (epoch+1, i, epoch_iters, loss.item(), cur_acc))
        
        acc = correct / total
        logging.info("%d epoch, cost time: %.5f, loss: %.5f, acc: %.3f " % (epoch+1, time.time()-start_epoch, total_loss, acc))


        if epoch % 5 == 0:
            torch.save(net.state_dict(), args.modeldir +'vgg_iter'+str(epoch)+'.pth')                        