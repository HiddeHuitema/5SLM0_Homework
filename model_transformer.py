import torch
from torch import nn

"""
Attempt to make a vision tranformer for segmentation
"""

# class PatchEmbeddings(nn.Module):
#     def __init__(self,batch_size,in_channels = 3,patch_size = 16,embedding_dim = 768,nr_patches = 14*14):
#         super().__init__()
#         self.in_channels = in_channels
#         self.patch_size = patch_size # Amount of patches in each dim, we have images of 1024*2048, we reduce them by factor 4 for training, this means each patch is 16*32 pixels
#         self.embedding_dim = 768
#         self.nr_patches = nr_patches
#         self.batch_size = batch_size
        

#         # A conv layer with kernel size of 16 and stride of 16 to create 16*16 patches
#         self.conv_patch = nn.Conv2d(in_channels=self.in_channels,out_channels=embedding_dim,kernel_size=patch_size,stride = patch_size)
#         self.flatten = nn.Flatten(start_dim=1,end_dim=2) # Flatten dims 1 and 2 (Heigth and width of image)

#         self.class_token_embeddings = nn.Parameter(torch.rand(batch_size,1,self.embedding_dim),requires_grad = True)
#         self.position_embeddings = nn.Parameter(torch.rand(batch_size,self.nr_patches,self.embedding_dim),requires_grad = True)
#         # self.embed_layer = nn.Linear(in_features=in_channels, out_features=embedding_dim)
    
#     def forward(self,x):
#         conv_x = self.conv_patch(x).permute(0,2,3,1)
#         flatten_x = self.flatten(conv_x)
#         # print(flatten_x.shape)
#         # print(self.class_token_embeddings.shape)
#         # print(self.position_embeddings.shape)
#         # output = torch.cat((self.class_token_embeddings,flatten_x),dim = 1)+self.position_embeddings
#         output = flatten_x+self.position_embeddings


#         return output
    

class ImageToPatch(nn.Module):
    def __init__(self,image_size,patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size= patch_size,stride = patch_size)

    def forward(self,x):
        x = self.unfold(x)
        x = x.permute(0,2,1)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels = 768,out_channels = 256):
        super().__init__()
        self.in_channels = 768
        self.out_channels = 256
        self.embedding = nn.Linear(in_features=in_channels,out_features=out_channels)

    def forward(self,x):
        x = self.embedding(x)
        return x
    
class VisionTransformerInput(nn.Module):
    def __init__(self,image_size,patch_size,in_channels,embedding_dim):
        super().__init__()
        self.imagetopatch = ImageToPatch(image_size,patch_size)
        self.patchembedding = PatchEmbedding(in_channels,embedding_dim)

        self.num_patches = (image_size//patch_size)**2

        self.position_embed = nn.Parameter(torch.randn(self.num_patches,embedding_dim))

    def forward(self,x):
        patches = self.imagetopatch(x)
        embedding = self.patchembedding(patches)

        output = embedding+ self.position_embed

        return output

class MSA_Block(nn.Module):
    def __init__(self,embedding_dim = 768, n_heads = 12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mha = nn.MultiheadAttention(num_heads=n_heads,embed_dim=embedding_dim)

    def forward(self,x):
        norm_x = self.layernorm(x)
        output, _ = self.mha(query = x,key = x,value = x)
        return output
    

class MLP_Block(nn.Module):
    def __init__(self,embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(nn.LayerNorm(embedding_dim),
                                 nn.Linear(embedding_dim,4*embedding_dim),
                                 nn.ReLU(),
                                 nn.Linear(4*embedding_dim,embedding_dim),
                                 nn.Dropout(0.2))
        

    def forward(self,x):
        output = self.mlp(x)
        return output
    

class SelfAttentionBlock(nn.Module):
    def __init__(self,embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mha = MSA_Block(embedding_dim=embedding_dim,n_heads=12)
        self.mlp= MLP_Block(embedding_dim=embedding_dim)

    def forward(self,x):
        x = x+self.mha(x)
        output = x+self.mlp(x)
        return output
    
class OutputProjection(nn.Module):
    def __init__(self,embedding_dim = 768, patch_size = 16, output_dims = 224,output_channels = 19, image_size = 224):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.projection = nn.Linear(embedding_dim,patch_size*patch_size*output_channels)
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
        # self.transconv = nn.ConvTranspose2d(embedding_dim,output_dims,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        # x = x.view(-1,self.output_dims,self.output_dims,self.embedding_dim)
        # output = self.transconv(x.permute(0,2,1).unsqueeze(2))
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,batch_size,num_blocks,in_channels = 3, image_size = 224,patch_size = 16,embedding_dim = 768,nr_patches = 14*14):
        super().__init__()
        
        
        
        
        
        self.embeddinglayer = VisionTransformerInput(image_size,patch_size,patch_size*patch_size*in_channels,embedding_dim)
        self.msa_bloc = MSA_Block()
        self.mlpblock = MLP_Block(embedding_dim)
        self.selfattentionblock = SelfAttentionBlock(embedding_dim)
        self.projectionblock = OutputProjection()
        heads = [ SelfAttentionBlock(embedding_dim) for i in range(num_blocks) ]
        # self.layers = nn.Sequential(
        #     nn.BatchNorm2d(num_features=in_channels),
        #     VisionTransformerInput(image_size, patch_size, in_channels, embedding_dim),
        #     nn.Sequential(*heads),
        #     OutputProjection(),
        # )
        self.attentionblocks = nn.Sequential(*heads)


    def forward(self,x):
        x = self.embeddinglayer(x)
        x = self.attentionblocks(x)
        output = self.projectionblock(x)
        return output
