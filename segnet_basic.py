import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNetBasic_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNetBasic_Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = (7,7),
                              padding = 3)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size = (2,2),
                                 stride=2,
                                 return_indices = True)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y, indices = self.pool(y)
        return y, indices, x.size()

class SegNetBasic_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNetBasic_Upsample, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size = (2,2))
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = (7,7),
                              padding = 3)

        self.bn = nn.BatchNorm2d(num_features = out_channels)

    def forward(self, x, indices, output_shape):
        y = self.unpool(input = x, indices = indices, output_size = output_shape)
        y = self.conv(y)
        y = self.bn(y)
        return y


class SegNetBasic(nn.Module):

    def __init__(self, in_channels, num_classes, depth = 4, final_batch_norm = True):
        super(SegNetBasic, self).__init__()

        self.final_batch_norm = final_batch_norm
        self.LRN = nn.LocalResponseNorm(size = 5, alpha = 1e-4, beta = 0.75)

        self.down = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                current_in = in_channels
            else:
                current_in = 64
            self.down.append(SegNetBasic_Downsample(in_channels = current_in, out_channels = 64))
    
        self.up = nn.ModuleList()
        for i in range(depth):
            self.up.append(SegNetBasic_Upsample(in_channels = 64, out_channels = 64))

        self.conv_classifier = nn.Conv2d(in_channels = 64,
                                         out_channels = num_classes,
                                         kernel_size = (1,1))
        if final_batch_norm:
            self.bn = nn.BatchNorm2d(num_features = num_classes)
        #self.softmax = nn.Softmax2d()
      
    def forward(self, x):
        y = self.LRN(x)
        indices_array = []
        shapes_array = []
        for down in self.down:
            indices_array.append([])
            shapes_array.append([])
            y, indices_array[-1], shapes_array[-1] = down(y)

        indices_array.reverse()
        shapes_array.reverse()

        for (up, indices, shape) in zip(self.up, indices_array, shapes_array):
            y = up(y, indices, shape)

        y = self.conv_classifier(y)
        if self.final_batch_norm:
            y = self.bn(y)

        #print(y.shape)
        #y = self.softmax(y)

        return y

def main():
    net = SegNetBasic(in_channels = 4, num_classes = 3, final_batch_norm = False)

    print(net)

if __name__ == "__main__":
    main()





