import matplotlib.pyplot as plt
import torch

from Visualize import visualize_result

# Output or visualize the predictions, the reference, the color input and their diff images
def output_predictions(predictions, data_loader, title, output_filename):
        save_to_file = not output_filename is None
        if save_to_file:
            fig = plt.figure(figsize = (15,10))
            fig.suptitle(title)
        row_count = len(predictions)
        figure_row = 1
        for prediction in predictions:
            _, sample_index, inferred_image = prediction
            (light, albedo, _, _), reference_image = data_loader.dataset[sample_index]
            visualize_result(light * albedo, inferred_image, reference_image, figure_row, row_count, show=False)
            figure_row += 1
        if save_to_file:
            plt.savefig(output_filename)
            plt.close(fig)
        else:
            plt.show()


def analyze_dataset(model, data_loader, output_folder:str=None):
    device = next(model.parameters()).device

    model.eval()

    predictions = []
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            input, reference_images = batch
            batch_size = reference_images.shape[0]
            batch_index_offset = batch_size * batch_index
            
            light, albedo, normals, positions = input
            light = light.to(device)
            albedo = albedo.to(device)
            normals = normals.to(device)
            positions = positions.to(device)
            reference_images = reference_images.to(device)

            inferred_images = model.forward((light, albedo, normals, positions))

            for b in range(0, batch_size):
                loss = model.loss_function(inferred_images[b], reference_images[b]).item()
                predictions.append((loss, batch_index_offset + b, inferred_images[b]))

    predictions.sort()

    # Visualize or output the four worst and four best predictions
    worst_filename = None if output_folder is None else output_folder + "/WorstPrediction.png"
    best_filename = None if output_folder is None else output_folder + "/BestPrediction.png"
    output_predictions(predictions[-5:-1], data_loader, "Worst predictions", worst_filename)
    output_predictions(predictions[0:4], data_loader, "Best predictions", best_filename)


# Debug analyze code
if __name__ == '__main__':
    from GlobalIlluminationFiltering import GlobalIlluminationFiltering
    from Visualize import visualize_result
    from ImageDataset import ImageDataset

    from torch.utils.data import DataLoader

    partial_set = True
    validation_set = ImageDataset(["Dataset/san-miguel/inputs"], partial_set=partial_set)
    validation_data_loader = DataLoader(validation_set, batch_size=8)

    model = GlobalIlluminationFiltering()

    analyze_dataset(model, validation_data_loader)