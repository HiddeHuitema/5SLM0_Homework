"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import collections
import matplotlib.pyplot as plt


from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from torch import nn
from torchvision.transforms import v2




def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./Datasets/CityScapes", help="Path to the data")
    parser.add_argument("--epochs",type = int, default = 1, help = "Amount of epochs for training")
    parser.add_argument("--batch_size",type = int, default = 1, help = "Batch size for training")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    transforms = v2.Compose([v2.Resize((1024,2048)),v2.ToImage(),v2.ToDtype(torch.float32,scale = True),v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    target_transforms = v2.Compose([v2.Resize((1024,2048)),v2.ToImage(),v2.ToDtype(torch.float32,scale = True)])
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',transform = transforms,target_transform=target_transforms)

    # train_data = dataset[:int(len(dataset)*0.9)]
    # val_data = dataset[int(len(dataset)*0.9):]
    trainloader = torch.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True)

    # visualize example images
    sample = dataset[0]
    img, target = sample
    plt.imshow(img[0,:,:])
    plt.show()
    plt.imshow(target[0,:,:])
    plt.show()
    print(f"{type(img) = } \n {type(target) = }")
    print(f"{torch.Tensor.size(img) = }\n {torch.Tensor.size(target) = }")

    # define model
    model = Model()#.cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (data,labels) in enumerate(trainloader):
            data = data.view(args.batch_size,-1,1024,2048)#.cuda()
            labels = labels.view(args.batch_size,-1,1024,2048)#.cuda()
            output = model.forward(data)
            loss = criterion(output,labels.long().squeeze()) 

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx == 1:
                print('im training')
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
