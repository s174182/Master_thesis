import torch
import torchvision.transforms.functional as TF
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)    

# Create the double convolution class used in Unet
class twoConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(twoConv, self).__init__()
        self.conv = nn.Sequential(
            # Convolution, batchnorm, relu times two
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    # Create forward pass where we simply do the 2x convolutions on x
    def forward(self, x):
        return self.conv(x)

    
# Define network class
class Unet(nn.Module):
    def __init__(
            self, in_channels=1, out_channels = 1, features=[64, 128, 256, 512]
            ):
        super(Unet,self).__init__() # Inherit and initialize
        
        # Create module list for ups and downs of unet
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        # Maxpool
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Down part of Unet
        for feature in features:
            # Append two convolutions and have feature as out channels
            self.downs.append(twoConv(in_channels, out_channels = feature))
            in_channels = feature
            
        # Up part of Unet
        for feature in reversed(features): 
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size = 2, stride = 2
                    )
                )
            self.ups.append(twoConv(feature*2, feature))
            
        # Bottleneck layer (bottom part of unet)
        self.bottleneck = twoConv(features[-1], features[-1]*2)
        
        # Final convolution to out
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Sigmoid output
        #self.sigmoid = nn.Sigmoid()
                    
    # Define forward pass
    def forward(self, x):
        # List skip connections
        skip_connections = []
        x=x.float()
        
        # Go down
        for down in self.downs:
            # Compute down, and append to skip_connections
            x = down(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        
        # Compute bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
            
        # Go up
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Take the up layer at index
            skip_connection = skip_connections[idx//2] # Take appropriate skip connection


            # Resize to always ensure correct output size
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])
            
            # Concatenate the skip_connection
            conc_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx+1](conc_skip)
        
        # Return final convolution of x after it is activated with sigmoid
        x = self.final_conv(x)
        return x

def test2():
    x = torch.randn((1,1,256,256))
    model = Unet(in_channels=1, out_channels=1)
    preds = model(x)
    
    assert preds.shape == x.shape
    
    print(preds.shape)
 
if __name__ == "__main__":
    test2()
 
 
 
 
 
 
 
 
