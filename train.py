"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import collections
import matplotlib.pyplot as plt
import wandb
import utils

from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from torch import nn
from torchvision.transforms import v2
# import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR

from helpers import *




def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./Datasets/CityScapes", help="Path to the data")
    parser.add_argument("--epochs",type = int, default = 50, help = "Amount of epochs for training")
    parser.add_argument("--batch_size",type = int, default = 20, help = "Batch size for training")
    parser.add_argument("--resizing_factor" ,type = int, default = 16, help = "Resizing factor for the size of the images, makes training on cpu faster for testing purposes")
    parser.add_argument("--n_workers", type = int, default = 1, help = "Number of workers for dataloading" )
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    learning_rate = 0.01
    wandb.init(
        project = '5SLM0 first test',

        config = {
            'learning rate':learning_rate,
            'architecture': 'CNN',
            'dataset': 'Cityscapes',
            'epochs': args.epochs,
        }
    )
    # data loading
    transforms = v2.Compose([v2.Resize((1024//args.resizing_factor,2048//args.resizing_factor)),v2.ToImage(),v2.ToDtype(torch.float32,scale = True),v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    target_transforms = v2.Compose([v2.Resize((1024//args.resizing_factor,2048//args.resizing_factor)),v2.ToImage()])

    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',transform = transforms,target_transform=target_transforms)

    indices_train = range(0,int(0.9*len(dataset)))
    indices_val = range(int(0.9*len(dataset)),len(dataset))
    trainset = torch.utils.data.Subset(dataset,indices_train)
    valset = torch.utils.data.Subset(dataset,indices_val)

    trainloader = torch.utils.data.DataLoader(trainset,batch_size = args.batch_size,shuffle = True,num_workers = args.n_workers)
    valloader = torch.utils.data.DataLoader(valset,batch_size = args.batch_size,shuffle = True,num_workers = args.n_workers)


    model = Model().cuda()
    # model.load_state_dict(torch.load('model_noise2.pth'))
    # define optimizer and loss function (don't forget to ignore class index 255)
    weights = [[1.0],
                [1.8088044976264959],
                [1.1562662699888433],
                [1.9799768118478331],
                [1.9824706319865246],
                [1.9748631680266007],
                [1.9871467635027236],
                [1.9715065736223831],
                [1.6200170629798962],
                [1.9714698225887604],
                [1.9277544680944152],
                [1.947933410627174],
                [1.9944619692428849],
                [1.8546759127600463],
                [1.9997051167064073],
                [1.9859173539255792],
                [2.0],
                [1.9990645986918383],
                [1.9934443156213768]]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights, ignore_index=255)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = ExponentialLR(optimizer,gamma = 0.9)
    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (data,target) in enumerate(trainloader):
            data = data.cuda()
            target = (target).squeeze(dim = 1).long()
            target = utils.map_id_to_train_id(target).cuda()
            output = model.forward(data)
        

            loss = criterion(output,target) 


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(loss)
        epoch_loss = running_loss/len(trainloader)
        epoch_data['loss'].append(epoch_loss)
        

        model.eval()
        running_loss = 0
        for batch_idx, (data,target) in enumerate(valloader):
            data = data.cuda()
            target = (target).long().squeeze(dim = 1)
            target = utils.map_id_to_train_id(target).cuda()
            

            output = model.forward(data)
            running_loss += criterion(output,target).item()

            # del data, target, output
            # torch.cuda.empty_cache()

        scheduler.step()
        validation_loss = running_loss/len(valloader)
        epoch_data['validation_loss'].append(validation_loss)
        print("Epoch {}/{}, Loss = {:6f}, Validation loss = {:6f},lr = {:6f}".format(epoch,args.epochs,epoch_loss,validation_loss,optimizer.param_groups[0]['lr']))
        wandb.log({'loss': epoch_loss, 'val_loss': validation_loss})

    torch.save(model.state_dict(),'model_baseline2.pth')


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
