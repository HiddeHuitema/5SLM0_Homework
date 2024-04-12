import torch
# import collections
# import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

# from model import Model
# from torchvision.datasets import Cityscapes
# from argparse import ArgumentParser
from torch import nn
# from torchvision.transforms import v2

# class diceloss(nn.Module):
#     def __init__(self, num_classes = 20):
#         super().__init__()
#         self.num_classes = num_classes

#     def forward(self, output, targets):
#         probabilities = nn.Softmax(dim = 1)(output)
#         targets_one_hot = torch.nn.functional.one_hot(targets,self.num_classes)
#         # print(targets_one_hot.shape)
#         targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
#         # print(targets_one_hot.shape)

#         intersection = (targets_one_hot * probabilities).sum()


#         mod_a = intersection.sum()
#         mod_b = targets.numel()

#         dice_coefficient = 2. * intersection / (mod_a + mod_b + 1e-6)
#         dice_loss = -dice_coefficient.log()
#         return dice_loss


# def mask_to_rgb(mask, class_to_color):
#     """
#     Converts a numpy mask with multiple classes indicated by integers to a color RGB mask.

#     Parameters:
#         mask (numpy.ndarray): The input mask where each integer represents a class.
#         class_to_color (dict): A dictionary mapping class integers to RGB color tuples.

#     Returns:
#         numpy.ndarray: RGB mask where each pixel is represented as an RGB tuple.
#     """
#     # Get dimensions of the input mask
#     height, width = mask.shape

#     # Initialize an empty RGB mask
#     rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

#     # Iterate over each class and assign corresponding RGB color
#     for class_idx, color in class_to_color.items():
#         # Mask pixels belonging to the current class
#         class_pixels = mask == class_idx
#         # Assign RGB color to the corresponding pixels
#         rgb_mask[class_pixels] = color

#     return rgb_mask

# def visualize_segmentation(model, dataloader, num_examples=5):
#     """
#     Visualizes segmentation results from a given model using a dataloader.

#     Args:
#         model (torch.nn.Module): The segmentation model to visualize.
#         dataloader (torch.utils.data.DataLoader): Dataloader providing image-mask pairs.
#         num_examples (int, optional): Number of examples to visualize. Defaults to 5.

#     Returns:
#         None
#     """
#     model.eval()
#     with torch.no_grad():
#         for i, (images, masks) in enumerate(dataloader):
#             if i >= num_examples:
#                 break
            
#             outputs = model(images.cuda())
#             outputs = torch.softmax(outputs, dim=1)
#             predicted = torch.argmax(outputs, 1)

#             images = images.numpy()
#             masks = masks.numpy()

#             predicted = predicted.cpu().numpy()

#             for j in range(images.shape[0]):
#                 image = renormalize_image(images[j].transpose(1, 2, 0))
#                 # print(np.unique(image))

#                 mask = masks[j].squeeze()
#                 pred_mask = predicted[j]
                                
#                 # Convert mask and predicted mask to RGB for visualization
#                 mask_rgb = mask_to_rgb(mask, colors)
#                 pred_mask_rgb = mask_to_rgb(pred_mask, colors)
                
#                 # Get unique classes present in the ground truth and predicted masks
#                 unique_classes_gt = np.unique(mask)
#                 unique_classes_pred = np.unique(pred_mask)
                
#                 unique_classes_gt = np.delete(unique_classes_gt, [0, -1])
#                 unique_classes_pred= np.delete(unique_classes_pred, 0)
                
#                 unique_classes_gt[unique_classes_gt == 255] = 0
#                 unique_classes_pred[unique_classes_pred == 255] = 0
                
                
#                 # Map class indices to class names from the VOC2012 dataset
#                 classes_gt = [class_names[int(idx)] for idx in unique_classes_gt]
#                 classes_pred = [class_names[int(idx)] for idx in unique_classes_pred]
                
#                 plt.figure(figsize=(10, 5))
#                 plt.subplot(1, 3, 1)
#                 plt.imshow(image)
#                 plt.title('Image')
#                 plt.axis('off')

#                 plt.subplot(1, 3, 2)
#                 plt.imshow(mask_rgb)
#                 plt.title('Ground Truth')
#                 plt.axis('off')

#                 plt.subplot(1, 3, 3)
#                 plt.imshow(pred_mask_rgb)
#                 plt.title('Predicted Mask ')
#                 plt.axis('off')

#                 plt.show()



# def renormalize_image(image):
#     """
#     Renormalizes the image to its original range.
    
#     Args:
#         image (numpy.ndarray): Image tensor to renormalize.
    
#     Returns:
#         numpy.ndarray: Renormalized image tensor.
#     """
#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]  
#     renormalized_image = image * std + mean
#     return renormalized_image       


# # visualize examples (No modifications needed)
# colors = {
#         0: (255,0,255),
#         1: (128,0,0),
#         2: (255,0,0),  
#         3: (255,140,0),   
#         4: (214, 39, 40), 
#         5: (148, 103, 189),
#         6: (140, 86, 75),
#         7: (227, 119, 194),
#         8: (127, 127, 127),
#         9: (188, 189, 34),
#         10: (23, 190, 207),
#         11: (174, 199, 232),
#         12: (255, 187, 120),
#         13: (152, 223, 138),
#         14: (255, 152, 150),
#         15: (197, 176, 213),
#         16: (196, 156, 148),
#         17: (247, 182, 210),
#         18: (199, 199, 199),
#         20: (255,215,0),
#         21: (240,230,140),
#         22: (107,142,35),
#         23: (173,255,47),
#         24: (0,128,128),
#         25: (0,206,209),
#         26: (30,144,255),
#         27: (25,25,112),
#         28: (138,43,226),
#         29: (139,0,139),
#         30: (255,20,147),
#         31: (219, 219, 141),
#         32: (199,59,0),
#         33: (0, 0, 0)
# }

# class_names = [
#     'road','sidewalk','parking','rail track','person','rider','car','truck','bus','on rails',
#     'motorcycle', 'bicycle','caravan','trailer', 'building', 'wall', 'fence' , 'guard rail', 'bridge',
#     'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky',
#     'ground', 'dynamic','static','.','.','.','.','.','.','.','.','.','.','.','.','.','.'
# ]

class BlendGradient(object):
    def __init__(self,alpha_range):
        self.alpha_range = alpha_range

    def __call__(self,tensor):
        gradients = self.gradient_batch(tensor)
        blended = self.blend_images(tensor,gradients)
        return blended.squeeze(0)



    def gradient_batch(self,batch):
        batch = batch.unsqueeze(0)
        ten=torch.unbind(batch,dim = 1)
        # separate rbg channels
        r = ten[0].unsqueeze(0).permute(1,0,2,3)#
        g = ten[1].unsqueeze(0).permute(1,0,2,3)#.unsqueeze(0)
        b = ten[2].unsqueeze(0).permute(1,0,2,3)#.unsqueeze(0)

        convx =np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
        conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight=nn.Parameter(torch.from_numpy(convx).float().unsqueeze(0).unsqueeze(0))
        G_x_r=conv1(Variable(r)).data.view(-1,1,r.shape[2],r.shape[3])
        G_x_g=conv1(Variable(g)).data.view(-1,1,g.shape[2],g.shape[3])
        G_x_b=conv1(Variable(b)).data.view(-1,1,b.shape[2],b.shape[3])
        

        convy=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
        conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight=nn.Parameter(torch.from_numpy(convy).float().unsqueeze(0).unsqueeze(0))
        G_y_r=conv2(Variable(r)).data.view(-1,1,r.shape[2],r.shape[3])
        G_y_g=conv2(Variable(g)).data.view(-1,1,g.shape[2],g.shape[3])
        G_y_b=conv2(Variable(b)).data.view(-1,1,b.shape[2],b.shape[3])

        G_r = torch.sqrt(torch.pow(G_x_r,2)+ torch.pow(G_y_r,2))
        G_g = torch.sqrt(torch.pow(G_x_g,2)+ torch.pow(G_y_g,2))
        G_b = torch.sqrt(torch.pow(G_x_b,2)+ torch.pow(G_y_b,2))
        return (G_r+G_g+G_b)/3
    

    def blend_images(self,img,edge):
        edge = edge.repeat(1,3,1,1)
        alpha = (self.alpha_range[0]-self.alpha_range[1])*torch.randn(1)+self.alpha_range[1]
        blend = (alpha)*img + (1-alpha)*edge
        return blend
    

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


    

# def paintbynr(tensor,target,alpha):
#     target_norm = norm_image(target)
#     target_norm = target_norm.unsqueeze(1)
#     target_norm = target_norm.repeat(1,3,1,1)
#     return alpha*tensor + (1-alpha)*target_norm
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