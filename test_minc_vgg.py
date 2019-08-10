import numpy as np
import os, time

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets.minc2500 import MINC2500




if __name__ == "__main__":
    ## create test dataset
    test_folder = "./data/minc-2500/"
    model_path  = "./model/vgg_iter145.pth"
    test_split  = 1

    test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    test_set    = MINC2500(root_dir=test_folder, set_type='test', split=test_split, transform=test_trans)
    test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    
    ## define model
    net = models.vgg19()
    net.classifier._modules['6'] = nn.Linear(4096, 23)
    net.load_state_dict(torch.load(model_path))
    net.cuda()


    ## test dataset
    total_cnt = 0
    correct   = 0
    for i, (images, labels) in enumerate(test_loader):
        images        = Variable(images, requires_grad=False).type(torch.cuda.FloatTensor)
        outputs       = net(images)
        _, predicted  = torch.max(outputs.data, 1)
        correct      += (predicted.cpu() == labels.cpu()).sum().numpy().astype('float')
        total_cnt    += labels.cpu().size()[0]

    acc = correct / total_cnt
    print("test dataset %d, count: %d , accuray: %.3f" % (test_split, total_cnt, acc))