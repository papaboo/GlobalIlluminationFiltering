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

    red_channel = np.fromstring(exr_data.channels('R')[0], dtype=np.float32).reshape(height, width)
    green_channel = np.fromstring(exr_data.channels('G')[0], dtype=np.float32).reshape(height, width)
    blue_channel = np.fromstring(exr_data.channels('B')[0], dtype=np.float32).reshape(height, width)

    image_matrix = np.stack((red_channel, green_channel, blue_channel), axis=0)

    return torch.Tensor(image_matrix)


class ImageDataset(Dataset):
    def __init__(self, image_roots:List[str], partial_set=False):
        super().__init__()
        self.images = []
        
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
                        if not partial_set or image_number[-1] == '0':
                            images = self.load_images(image_root, image_number)
                            self.images.append(images)
                    
    
    def __len__(self):
        return len(self.images)


    # Return tuple of input images and target image
    # ((color, albedo, shading_normal, world_position), reference)
    def __getitem__(self, index):
        return self.images[index]


    def load_images(self, directory:str, number:str):
        make_path = lambda name: directory + "/" + name + number + ".exr"

        color_matrix = load_exr_as_tensor(make_path("color"))
        albedo_matrix = load_exr_as_tensor(make_path("albedo"))
        normal_matrix = load_exr_as_tensor(make_path("shading_normal"))
        position_matrix = load_exr_as_tensor(make_path("world_position"))
        reference_matrix = load_exr_as_tensor(make_path("reference"))

        return ((color_matrix, albedo_matrix, normal_matrix, position_matrix), reference_matrix)

        
if __name__ == '__main__':
    training_set = ImageDataset(["Dataset/san-miguel/inputs", "Dataset/sponza/inputs"], partial_set=True)
    print("||training_set||", len(training_set))

    (color_tensor, albedo_tensor, normal_tensor, position_tensor), reference_tensor = training_set[0]
    show_data(reference_tensor, color_tensor, albedo_tensor, normal_tensor, position_tensor)