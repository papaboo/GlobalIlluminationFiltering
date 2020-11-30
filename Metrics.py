import math, torch
import torch.nn.functional as F

def MSE(noisy_tensor, reference_tensor):
    return ((noisy_tensor - reference_tensor) ** 2).mean()


# Peak signal-to-noise.
# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def PSNR(noisy_tensor, reference_tensor):
    max_value = torch.max(reference_tensor)
    mse = MSE(noisy_tensor, reference_tensor)
    return 10 * torch.log10(max_value * max_value / mse)


# Structural similarity index measure pr pixel in an image
# http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
# https://en.wikipedia.org/wiki/Structural_similarity
# https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
def SSIM_pr_pixel(noisy_tensor, reference_tensor, filter_std_dev = 1.5):
    L = torch.max(reference_tensor)

    kernel_halfsize = math.ceil(3 * filter_std_dev)
    kernel_size = 2 * kernel_halfsize + 1
    padding = kernel_size // 2

    # Tensor layout is [batch_size x channels x height x width]
    is_batched = len(noisy_tensor.size()) == 4
    batch_size = noisy_tensor.size()[0] if is_batched else 1
    channels = noisy_tensor.size()[-3]
    height = noisy_tensor.size()[-2]
    width = noisy_tensor.size()[-1]

    if not is_batched:
        noisy_tensor = noisy_tensor.unsqueeze(0)
        reference_tensor = reference_tensor.unsqueeze(0)

    # Create a gaussian filter window
    kernel_size = min(kernel_size, height, width)
    gaussian_1D = torch.Tensor([math.exp(-(x - kernel_size // 2) ** 2 / float(2 * filter_std_dev ** 2)) for x in range(kernel_size)])
    gaussian_1D = gaussian_1D / gaussian_1D.sum() # Normalize
    gaussian_1D = gaussian_1D.unsqueeze(0)

    gaussian_2D = gaussian_1D.t().mm(gaussian_1D)

    # Normalize weight near the border of the tensor.
    ones = torch.ones((1, 1, height, width), dtype=noisy_tensor.dtype)
    gaussian_2D = gaussian_2D.expand(1, 1, kernel_size, kernel_size)
    normalizer = F.conv2d(ones, gaussian_2D, padding=padding)

    gaussian_2D = gaussian_2D.expand(channels, 1, kernel_size, kernel_size)
    guassian_conv2d = lambda tensor: F.conv2d(tensor, gaussian_2D, padding=padding, groups=channels) / normalizer

    # calculating the mu parameter (locally) for both images using a gaussian filter calculates the luminosity params
    mu_x = guassian_conv2d(noisy_tensor)
    mu_y = guassian_conv2d(reference_tensor)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2 
    mu_xy = mu_x * mu_y

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma_x_sq = guassian_conv2d(noisy_tensor * noisy_tensor) - mu_x_sq
    sigma_y_sq = guassian_conv2d(reference_tensor * reference_tensor) - mu_y_sq
    sigma_xy = guassian_conv2d(noisy_tensor * reference_tensor) - mu_xy

    # Some constants for stability 
    c1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    c2 = (0.03) ** 2 

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_score = numerator / denominator
    return ssim_score if is_batched else ssim_score.squeeze(0)


def normalized_SSIM_pr_pixel(noisy_tensor, reference_tensor, filter_std_dev = 1.5):
    ssim_pr_pixel = SSIM_pr_pixel(noisy_tensor, reference_tensor, filter_std_dev)
    return ssim_pr_pixel * 0.5 + 0.5


# Structural similarity index measure.
# Result is in the range [-1, 1], where -1 is completely dissimilar and 1 is perfectly similar.
# http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
# https://en.wikipedia.org/wiki/Structural_similarity
# https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
def SSIM(noisy_tensor, reference_tensor, filter_std_dev = 1.5):
    return SSIM_pr_pixel(noisy_tensor, reference_tensor, filter_std_dev).mean()


if __name__ == '__main__':
    from ImageDataset import ImageDataset
    from Visualize import show_HDR_tensor

    training_set = ImageDataset(["Dataset/san-miguel/inputs"], partial_set=True)
    (color_tensor, _, _, _), reference_tensor = training_set[0]

    identity_psnr = PSNR(reference_tensor, reference_tensor).item()
    print(f"PSNR of same images: {identity_psnr:.6f}")

    noisy_psnr = PSNR(color_tensor, reference_tensor).item()
    print(f"PSNR of noisy and reference images: {noisy_psnr:.6f}")

    identity_ssim = SSIM(reference_tensor, reference_tensor).item()
    print(f"SSIM of same images: {identity_ssim:.6f}")

    noisy_ssim = SSIM(color_tensor, reference_tensor).item()
    print(f"SSIM of noisy and reference images: {noisy_ssim:.6f}")

    # Test that SSIM works on batched images. The mean of all channels should roughly equal the result from SSIM on the single image.
    stacked_color_tensor = torch.stack((color_tensor, color_tensor, color_tensor))
    stacked_reference_tensor = torch.stack((reference_tensor, reference_tensor, reference_tensor))
    ssim = SSIM(stacked_color_tensor, stacked_reference_tensor).item()
    print(f"SSIM of stacked noisy and reference images: {ssim:.6f}")

    # Test that SSIM works on a single channel image. The mean of all channels should roughly equal the result from SSIM on the full image.
    red, green, blue = torch.split(color_tensor, 1)
    red_ref, green_ref, blue_ref = torch.split(reference_tensor, 1)
    red_ssim = SSIM(red, red_ref).item()
    print(f"SSIM of red noisy and reference images: {red_ssim:.6f}")
    green_ssim = SSIM(green, green_ref).item()
    print(f"SSIM of green noisy and reference images: {green_ssim:.6f}")
    blue_ssim = SSIM(blue, blue_ref).item()
    print(f"SSIM of blue noisy and reference images: {blue_ssim:.6f}")
    print(f"  mean: {(red_ssim + green_ssim + blue_ssim) / 3:.6f}")

    # Visualize SSIM image
    ssim_image = normalized_SSIM_pr_pixel(color_tensor, reference_tensor, 1.5)
    show_HDR_tensor(ssim_image)