import torch
from torch import nn
import torch.nn.functional as F
from Utils import project_tensor, unproject_tensor

# Filter using basic (Conv2D, BatchNorm, Activation) blocks.
# First estimate a single value embedding pr pixel.
# Then feed the light and the embedding into a network that filters the light.
class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.name = "DepthNet"

        self.estimate_embedding = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

        self.filter_light = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=5, padding=2)
        )


    def forward(self, x):
        light, albedo, normals, positions = x

        is_batch = len(light.shape) == 4
        if not is_batch:
            light = light.unsqueeze(0)
            albedo = albedo.unsqueeze(0)
            normals = normals.unsqueeze(0)

        x = torch.cat((light, albedo, normals), dim=1)
        embedding = self.estimate_embedding(x)

        # Concatenate depth to the light
        light_embedding = torch.cat([light, embedding], dim=1)

        # TODO Filter one channel at a time to ensure the same weights pr color. Should reduce the number of free parameters.
        filtered_light = self.filter_light(light_embedding)

        # Multiply by albedo
        filtered_color = filtered_light * albedo

        if not is_batch:
            filtered_color = filtered_color.squeeze()

        return filtered_color



class DepthNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.name = "DepthNet"
        self.depth_size = 16

        self.estimate_depth = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Three dimensional convolution kernel for filtering the light
        # TODO Try with a gaussian instead of learning it. We simply start with the learned filter since it's simpler to set up.
        # self.conv_3D_padding = 2
        # self.conv_3D = torch.ones(5,5,5)
        self.light_filter = nn.Conv3d(4, 4, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        light, albedo, normals, positions = x

        if len(light.shape) == 3:
            light = light.unsqueeze(0)
            albedo = albedo.unsqueeze(0)
            normals = normals.unsqueeze(0)

        image_count = light.size()[0]
        height = light.size()[2]
        width = light.size()[3]

        x = torch.cat((light, albedo, normals), dim=1)
        embedding = self.estimate_embedding(x)

        # Add a channel of ones. It will be used after the gaussian filter to normalize the contribution.
        ones = torch.ones(image_count, 1, height, width)
        weighted_light = torch.cat([light, ones], dim=1)

        # Project images into 3D.
        depth = (embedding * (self.depth_size - 0.00001)).long().view(-1)
        depth = depth.repeat_interleave(4) # Extend to project the four channels in weighted_light.
        weighted_light_3D = project_tensor(weighted_light, 2, depth, self.depth_size)

        # Filter in 3D
        # filtered_light_3D = F.conv3d(weighted_light_3D, self.conv_3D, padding=self.conv_3D_padding)
        filtered_light_3D = self.light_filter(weighted_light_3D) # This uses an absolutely insane amount of memory and cannot work

        # Unproject
        filtered_light = unproject_tensor(filtered_light_3D, 2, depth)

        # Normalize
        filtered_light, weight = torch.split(filtered_light, dim=1)
        filtered_light = filtered_light / weight

        # Multiply by albedo
        filtered_color = filtered_light * albedo

        return filtered_color


if __name__ == "__main__":
    from ImageDataset import ImageDataset
    from Visualize import visualize_result

    # TODO Test with CUDA tensors
    dataset = ImageDataset(["Dataset/san-miguel/inputs"], partial_set=True)
    x, reference = dataset[0]
    (light, albedo, normals, positions) = x

    net = ConvNet()
    inferred = net.forward(x)

    visualize_result(light * albedo, inferred, reference)
