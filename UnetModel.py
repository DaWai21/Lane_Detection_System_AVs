import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),#(in,out,kernel,stride,pad,bias)
            nn.BatchNorm2d(out_channels),#Normalization
            nn.ReLU(inplace = True),#Activation
            # Second convolution
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    # forward the output of convo layer to maxpooling
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[64,128,256,512]): #(rgb, grey, filter or feature maps)
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size =2, stride = 2)

        # Down part (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part (Decoder)
        for feature in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(
            nn.ConvTranspose2d( feature*2, feature, kernel_size=2, stride=2)#(1024, 512)
            )
            # DoubleConv after concatenation
            self.ups.append(DoubleConv(feature*2, feature))
        
        #bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final 1x1 convolution maps features[0] to the desired number of output channels
        self.final_conv = nn.Conv2d(features[0], out_channels = out_channels, kernel_size=1)
    
    def forward(self,x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse skip connections
        
        # Decoder path
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Transposed convolution (upsampling)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.concat((skip_connection,x), dim=1)
            x = self.ups[i+1](concat_skip) # DoubleConv after concatenation

        return self.final_conv(x)

def test():
    # Test with 3 input channels (RGB) and 4 output channels (Multi-class)
    x = torch.randn((2, 3, 160, 160)) 
    model = UNET(in_channels=3, out_channels=4)
    pred = model(x)
    # Expected output shape: (Batch Size, Num Classes, Height, Width)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {pred.shape}")
    assert pred.shape == (2, 4, 160, 160)

if __name__ == '__main__':
    test()
