import matplotlib.pyplot as plt
import torch
import torchvision

def show_HDR_tensor(hdr_tensor):
    hdr_tensor_cpu = hdr_tensor.cpu()

    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_image(hdr_tensor_cpu))
    plt.title("HDR")

    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_image(to_sRGB(hdr_tensor_cpu)))
    plt.title("HDR sRGB")

    plt.show()


def show_data(reference_tensor, color_tensor, albedo_tensor, normal_tensor, position_tensor):
    
    nrows = 2
    ncols = 3

    albedo_tensor_cpu = albedo_tensor.cpu()
    reference_tensor_cpu = reference_tensor.cpu()

    plt.subplot(nrows, ncols, 1)
    plt.imshow(tensor_to_image(to_sRGB(reference_tensor_cpu)))
    plt.title("Reference")

    plt.subplot(nrows, ncols, 2)
    plt.imshow(tensor_to_image(to_sRGB(color_tensor.cpu())))
    plt.title("1 sample")

    plt.subplot(nrows, ncols, 3)
    rho = reference_tensor_cpu / (albedo_tensor_cpu + 0.00001)
    plt.imshow(tensor_to_image(to_sRGB(rho)))
    plt.title("Incoming light (reference / albedo)")

    plt.subplot(nrows, ncols, 4)
    plt.imshow(tensor_to_image(albedo_tensor_cpu))
    plt.title("Albedo")

    plt.subplot(nrows, ncols, 5)
    plt.imshow(tensor_to_image(normal_tensor.cpu()))
    plt.title("Normals")

    plt.subplot(nrows, ncols, 6)
    plt.imshow(tensor_to_image(position_tensor.cpu()))
    plt.title("Positions")

    plt.show()


def visualize_result(single_sample_tensor, infered_tensor, reference_tensor):
    plt.subplot(1, 4, 1)
    plt.imshow(tensor_to_image(to_sRGB(single_sample_tensor.cpu())))
    plt.title("1 sample")

    infered_tensor_cpu = infered_tensor.cpu()
    plt.subplot(1, 4, 2)
    plt.imshow(tensor_to_image(to_sRGB(infered_tensor_cpu)))
    plt.title("Inferred")

    reference_tensor_cpu = reference_tensor.cpu()
    plt.subplot(1, 4, 3)
    plt.imshow(tensor_to_image(to_sRGB(reference_tensor_cpu)))
    plt.title("Reference")

    diff_tensor_cpu = torch.abs(infered_tensor_cpu - reference_tensor_cpu)
    plt.subplot(1, 4, 4)
    plt.imshow(tensor_to_image(to_sRGB(diff_tensor_cpu)))
    plt.title("Diff")

    plt.show()


def tensor_to_image(tensor):
    to_image = torchvision.transforms.ToPILImage()
    clamped_tensor = torch.clamp(tensor, min=0.0, max=1.0)
    return to_image(clamped_tensor)


def to_sRGB(linear_tensor):
    return linear_tensor.pow(1.0 / 2.2) # TODO Proper linear to sRGB. But this is fine for now
