import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset
from typing import List

from Visualize import show_HDR_tensor, show_data

def load_exr_as_tensor(filename):
    import OpenEXR
    import Imath
    if not OpenEXR.isOpenExrFile(filename):
        raise Exception(f"File {filename} is not an EXR file.")
    exr_data = OpenEXR.InputFile(filename)

    exr_header = exr_data.header()
    dw = exr_header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    red_channel = np.frombuffer(exr_data.channels('R')[0], dtype=np.float32).reshape(height, width)
    green_channel = np.frombuffer(exr_data.channels('G')[0], dtype=np.float32).reshape(height, width)
    blue_channel = np.frombuffer(exr_data.channels('B')[0], dtype=np.float32).reshape(height, width)

    image_tensor = np.stack((red_channel, green_channel, blue_channel), axis=0)

    return torch.Tensor(image_tensor)


class ImageDataset(Dataset):
    def __init__(self, image_roots:List[str], mirror_images=True, swizzle_colors=True):
        super().__init__()
        self.images = []
        self.mirror_images = mirror_images
        self.swizzle_colors = swizzle_colors

        # Search for input/colorN.exr images and load all images associated with a sample.
        # NOTE This just barely fits in memory right now. At some point it should probably be loaded on the fly or stored compressed,
        # but right now it takes longer to load than to train, so keeping it in memory is a net win.
        for image_root in image_roots:
            if not os.path.isabs(image_root):
                image_root = os.getcwd() + "/" + image_root
            for (_, _, filenames) in os.walk(image_root):
                # Sort files by natural sorting, to get the proper temporal ordering
                filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
                for filename in filenames:
                    if filename.startswith("color"):
                        image_number = filename[5:-4] # Remove 'color' prefix and .exr extension from name
                        self.images.append((image_root, image_number, None))


    def __len__(self):
        return len(self.images)


    # Return tuple of input images and target image
    # ((color, albedo, shading_normal, world_position), reference)
    def __getitem__(self, index):
        image_root, image_number, images = self.images[index]
        if images is None:
            images = self.load_images(image_root, image_number)

            if self.mirror_images:
                capped_index = index % 4

                def flip_images(images, dims):
                    ((light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor) = images

                    light_tensor = torch.flip(light_tensor, dims)
                    albedo_tensor = torch.flip(albedo_tensor, dims)
                    normal_tensor = torch.flip(normal_tensor, dims)
                    position_tensor = torch.flip(position_tensor, dims)
                    reference_tensor = torch.flip(reference_tensor, dims)

                    return ((light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor)

                # Horizontal mirroring
                if capped_index == 1 or capped_index == 3:
                    images = flip_images(images, [2])

                # Vertical mirroring
                if capped_index == 2 or capped_index == 3:
                    images = flip_images(images, [1])

            if self.swizzle_colors:
                permutations = [[0,1,2], [1,0,2], [2,0,1], [0,2,1], [1,2,0], [2,1,0]]
                permutation = permutations[index % 6]

                ((light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor) = images

                light_tensor = light_tensor[permutation,:,:]
                albedo_tensor = albedo_tensor[permutation,:,:]
                normal_tensor = normal_tensor[permutation,:,:]
                position_tensor = position_tensor[permutation,:,:]
                reference_tensor = reference_tensor[permutation,:,:]

                images = ((light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor)

            self.images[index] = (image_root, image_number, images)

        return images


    def load_images(self, directory:str, number:str):
        make_path = lambda name: directory + "/" + name + number + ".exr"

        light_tensor = load_exr_as_tensor(make_path("color")) # The 'color' images only contain the incoming light, not the contribution from the BRDF
        albedo_tensor = load_exr_as_tensor(make_path("albedo"))
        normal_tensor = load_exr_as_tensor(make_path("shading_normal"))
        position_tensor = load_exr_as_tensor(make_path("world_position"))
        reference_tensor = load_exr_as_tensor(make_path("reference"))

        return ((light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor)


if __name__ == '__main__':
    training_set = ImageDataset(["Dataset/san-miguel/inputs", "Dataset/sponza/inputs"])
    print("||training_set||", len(training_set))

    (light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor = training_set[0]
    show_data(reference_tensor, light_tensor * albedo_tensor, albedo_tensor, normal_tensor, position_tensor)

    (light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor = training_set[1]
    show_data(reference_tensor, light_tensor * albedo_tensor, albedo_tensor, normal_tensor, position_tensor)

    (light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor = training_set[2]
    show_data(reference_tensor, light_tensor * albedo_tensor, albedo_tensor, normal_tensor, position_tensor)

    (light_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor = training_set[3]
    show_data(reference_tensor, light_tensor * albedo_tensor, albedo_tensor, normal_tensor, position_tensor)