"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import collections

from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from torch import nn


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    parser.add_argument("epochs",type = int, default = 5, help = "Amount of epochs for training")
    parser.add_argument("batch_size",type = int, default = 64, help = "Batch size for training")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    transforms = torch.transforms.Compose([torch.transforms.ToTensor(),torch.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',transform = transforms)
    trainloader = torch.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True)

    # visualize example images

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (data,labels) in enumerate(trainloader):
            output = model.forward(data)
            loss = criterion(output,labels) 

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_data['loss'].append(running_loss/len(trainloader))
  
    # save model
    torch.save(model,'/home/testmodel.pt')

    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
