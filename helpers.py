import torch

import numpy as np
from torch.autograd import Variable


from torch import nn



    
class BlendGradient(nn.Module):
    def __init__(self,alpha_range):
        self.alpha_range = alpha_range

    def forward(self,tensor):
        gradients = self.gradient_batch(tensor)
        blended = self.blend_images(tensor,gradients)
        return blended.squeeze(0)



    def gradient_batch(self,batch):
        ten=torch.unbind(batch,dim = 1)
        # separate rbg channels
        r = ten[0].unsqueeze(0).permute(1,0,2,3)#
        g = ten[1].unsqueeze(0).permute(1,0,2,3)#.unsqueeze(0)
        b = ten[2].unsqueeze(0).permute(1,0,2,3)#.unsqueeze(0)

        convx =np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
        conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
        conv1.weight=nn.Parameter(torch.from_numpy(convx).float().unsqueeze(0).unsqueeze(0).cuda())
        G_x_r=conv1(Variable(r)).data.view(-1,1,r.shape[2],r.shape[3])
        G_x_g=conv1(Variable(g)).data.view(-1,1,g.shape[2],g.shape[3])
        G_x_b=conv1(Variable(b)).data.view(-1,1,b.shape[2],b.shape[3])
        

        convy=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
        conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
        conv2.weight=nn.Parameter(torch.from_numpy(convy).float().unsqueeze(0).unsqueeze(0).cuda())
        G_y_r=conv2(Variable(r)).data.view(-1,1,r.shape[2],r.shape[3])
        G_y_g=conv2(Variable(g)).data.view(-1,1,g.shape[2],g.shape[3])
        G_y_b=conv2(Variable(b)).data.view(-1,1,b.shape[2],b.shape[3])

        G_r = torch.sqrt(torch.pow(G_x_r,2)+ torch.pow(G_y_r,2))
        G_g = torch.sqrt(torch.pow(G_x_g,2)+ torch.pow(G_y_g,2))
        G_b = torch.sqrt(torch.pow(G_x_b,2)+ torch.pow(G_y_b,2))
        return (G_r+G_g+G_b)/3
    

    def blend_images(self,img,edge,alpha_range = torch.tensor([0.3,1]).cuda()):
        edge = edge.repeat(1,3,1,1)
        # print(edge.shape)
        # print(torch.tensor(torch.rand(1)).cuda())

        alpha = (alpha_range[0]-alpha_range[1])*torch.rand(1).cuda()+alpha_range[1]

        batch_size = img.shape[0]
        split = batch_size//2
        img_to_transform = img[:split]
        edge_to_blend = edge[:split]
        img_no_transform = img[split:]
        blend = (alpha)*img_to_transform + (1-alpha)*edge_to_blend
        return torch.cat((blend,img_no_transform),0) 


   

# Creating a new transformer that can be used to add noise to the images
class AddGaussianNoise(object):
    '''
    A class to add gaussian noise to a tensor

    ...
    Attribures
    ----------
    std: float, optional
        desired standard deviation for the gaussian noise
    mean: float, optional
        desired mean of the gaussian noise
    
    Methods
    -------
    __call__(self,tensor):
        Returns the tensor with added noise
    '''
    def __init__(self,std = 0.3,mean =0.5):
        '''
        Constructs all the necessary attributes for the AddGaussianNoise object.

        Parameters
        ----------
        std: float, optional
            desired standard deviation for the gaussian noise
        mean: float, optional
            desired mean of the gaussian noise
        ''' 
        self.mean = mean
        self.std = std
        
    def __call__(self,tensor):
        '''
        Returns tensor with added noise

            Params:
                tensor (pytorch tensor): original tensor to add noise to

            Returns:
                tensor+added noise
        '''

        return tensor + torch.randn(tensor.size()) * self.std + self.mean  
    

def norm_image(img):
    img_norm = (img-torch.min(img))/(torch.max(img)-torch.min(img))
    return img_norm


    

def paintbynr(tensor,target,alpha):
    target_norm = norm_image(target)
    target_norm = target_norm.unsqueeze(1)
    target_norm = target_norm.repeat(1,3,1,1)
    batch_size = tensor.shape[0]
    split = batch_size//2
    tensor_no_transform = tensor[:split]
    tensor_to_transform = tensor[split:]
    targets_for_transform = target_norm[split:]

    transformed_tensor =  alpha*tensor_to_transform + (1-alpha)*targets_for_transform
    
    return torch.cat((tensor_no_transform,transformed_tensor),0)