"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import collections
import matplotlib.pyplot as plt
import wandb

from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from torch import nn
from torchvision.transforms import v2

from helpers import *




def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./Datasets/CityScapes", help="Path to the data")
    parser.add_argument("--epochs",type = int, default = 10, help = "Amount of epochs for training")
    parser.add_argument("--batch_size",type = int, default = 8, help = "Batch size for training")
    parser.add_argument("--resizing_factor" ,type = int, default = 16, help = "Resizing factor for the size of the images, makes training on cpu faster for testing purposes")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    learning_rate = 0.0005
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

    indices_train = range(0,int(0.01*len(dataset)))
    indices_val = range(int(0.99*len(dataset)),len(dataset))
    trainset = torch.utils.data.Subset(dataset,indices_train)
    valset = torch.utils.data.Subset(dataset,indices_val)



    trainloader = torch.utils.data.DataLoader(trainset,batch_size = args.batch_size,shuffle = True,num_workers=8)
    valloader = torch.utils.data.DataLoader(valset,batch_size = args.batch_size,shuffle = True,num_workers=8)

    # visualize example images
    print(dataset[0][0].size())
    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (data,target) in enumerate(trainloader):
            data = data.cuda()
            target = target.cuda()
            output = model.forward(data)
            loss = criterion(output,target.long().squeeze(dim = 1)) 


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss/len(trainloader)
        epoch_data['loss'].append(epoch_loss)
        
        running_loss = 0
        for batch_idx, (data,target) in enumerate(valloader):
            data = data.cuda()
            target = target.cuda()
            output = model.forward(data)
            running_loss += criterion(output,target.long().squeeze(dim = 1))

        validation_loss = running_loss/len(valloader)
        epoch_data['validation_loss'].append(validation_loss)
        print("Epoch {}/{}, Loss = {:6f}, Validation loss = {:6f}".format(epoch,args.epochs,epoch_loss,validation_loss))
        wandb.log({'loss': epoch_loss, 'val_loss': validation_loss})

    torch.save(model,'snellius_model.pt')


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
