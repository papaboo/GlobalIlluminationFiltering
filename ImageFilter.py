import math, torch


def gaussian_2D(image_tensor, std_dev:float):
    is_batched = len(image_tensor.size()) == 4
    if not is_batched:
        image_tensor = image_tensor.unsqueeze(0)

    height = image_tensor.size()[-2]
    width = image_tensor.size()[-1]

    kernel_halfsize = math.ceil(3 * std_dev)
    kernel_size = 2 * kernel_halfsize + 1

    gaussian_1D = torch.Tensor([math.exp(-(x - kernel_size // 2) ** 2 / float(2 * std_dev ** 2)) for x in range(kernel_size)])
    gaussian_1D = gaussian_1D.to(image_tensor.device)

    # Weight tensor used to normalized near the borders
    weight_tensor = torch.ones((1,1,height,width), dtype=torch.float).to(image_tensor.device)

    # Filter horizontally
    intermediate_tensor = torch.zeros_like(image_tensor)
    intermediate_weight_tensor = torch.zeros_like(weight_tensor)
    for x in range(-kernel_halfsize, kernel_halfsize+1):
        weight = gaussian_1D[x + kernel_halfsize]

        source_start = max(0, 0 - x)
        source_end = min(width, width - x)
        target_start = max(0, 0 + x)
        target_end = min(width, width + x)

        intermediate_tensor[:, :, 0:height, target_start:target_end] += weight * image_tensor[:, :, 0:height, source_start:source_end]
        intermediate_weight_tensor[:, :, 0:height, target_start:target_end] += weight * weight_tensor[:, :, 0:height, source_start:source_end]

    # Filter vertically
    result = torch.zeros_like(intermediate_tensor)
    result_weight = torch.zeros_like(intermediate_weight_tensor)
    for y in range(-kernel_halfsize, kernel_halfsize+1):
        weight = gaussian_1D[y + kernel_halfsize]

        source_start = max(0, 0 - y)
        source_end = min(height, height - y)
        target_start = max(0, 0 + y)
        target_end = min(height, height + y)

        result[:, :, target_start:target_end, 0:width] += weight * intermediate_tensor[:, :, source_start:source_end, 0:width]
        result_weight[:, :, target_start:target_end, 0:width] += weight * intermediate_weight_tensor[:, :, source_start:source_end, 0:width]

    # Normalize
    result /= result_weight

    if not is_batched:
        result = result.squeeze(0)

    return result


if __name__ == "__main__":
    from Visualize import show_HDR_tensor
    from ImageDataset import load_exr_as_tensor
    import torch.nn.functional as F

    def gaussian_2D_ref(image_tensor, std_dev:float):
        is_batched = len(image_tensor.size()) == 4
        channels = image_tensor.size()[-3]
        height = image_tensor.size()[-2]
        width = image_tensor.size()[-1]

        if not is_batched:
            image_tensor = image_tensor.unsqueeze(0)

        kernel_halfsize = math.ceil(3 * std_dev)
        kernel_size = 2 * kernel_halfsize + 1
        padding = kernel_size // 2

        kernel_size = min(kernel_size, height, width)
        gaussian_1D = torch.Tensor([math.exp(-(x - kernel_size // 2) ** 2 / float(2 * std_dev ** 2)) for x in range(kernel_size)])
        gaussian_1D = gaussian_1D / gaussian_1D.sum() # Normalize
        gaussian_1D = gaussian_1D.unsqueeze(0)

        gaussian_2D = gaussian_1D.t().mm(gaussian_1D)
        gaussian_2D = gaussian_2D.to(image_tensor.device)

        # Border normalizer
        ones = torch.ones((1, 1, height, width), dtype=image_tensor.dtype).to(image_tensor.device)
        gaussian_2D = gaussian_2D.expand(1, 1, kernel_size, kernel_size)
        border_normalizer = F.conv2d(ones, gaussian_2D, padding=padding)

        gaussian_2D = gaussian_2D.expand(channels, 1, kernel_size, kernel_size)
        result = F.conv2d(image_tensor, gaussian_2D, padding=padding, groups=channels) / border_normalizer

        if not is_batched:
            result = result.squeeze(0)

        return result


    tensor = load_exr_as_tensor('Dataset/classroom/inputs/reference0.exr')

    ref_guassian_blur = gaussian_2D_ref(tensor, 10)
    show_HDR_tensor(ref_guassian_blur)

    guassian_blur = gaussian_2D(tensor, 10)
    show_HDR_tensor(guassian_blur)

    assert torch.allclose(guassian_blur, ref_guassian_blur)