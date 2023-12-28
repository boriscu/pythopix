import os
import time
import torch
import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from typing import Optional, List

from .data_handling import export_to_csv
from .model_operations import process_image, segregate_images
from .utils import custom_sort_key
from .theme import console, INFO_STYLE, SUCCESS_STYLE
from .labels_operations import Label, read_yolo_labels


def evaluate_dataset(
    test_images_folder: str,
    model_path: Optional[str] = None,
    num_images: int = 100,
    verbose: bool = False,
    print_results: bool = False,
    copy_images: bool = False,
) -> List[dict]:
    """
    Main function to execute the YOLO model analysis script.

    Args:
    model_path (str): Path to the model weights file.
    test_images_folder (str): Path to the test images folder.
    num_images (int): Number of images to separate for additional augmentation.
    verbose (bool): Enable verbose output for model predictions.
    print_results (bool): Print the sorted image data results.
    copy_images (bool): Copy images to a separate folder for additional augmentation.

    Returns:
    List[dict]: A list of dictionaries containing sorted image data based on the evaluation.
    """

    start_time = time.time()

    images = [
        os.path.join(test_images_folder, file)
        for file in os.listdir(test_images_folder)
        if file.endswith(".jpg") or file.endswith(".png")
    ]

    if model_path is None or not os.path.exists(model_path):
        console.print(
            "Model path not provided or not found. Using default YOLO model.",
            style=INFO_STYLE,
        )
        model = YOLO("yolov8n")
    else:
        model = YOLO(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_data_list = []
    predictions_dict = {}

    for image_path in tqdm.tqdm(images, desc="Processing Images"):
        image_data, predictions = process_image(image_path, model, verbose=verbose)
        image_data_list.append(image_data)
        predictions_dict[image_path] = predictions

    sorted_image_data = sorted(image_data_list, key=custom_sort_key, reverse=True)

    if copy_images:
        segregate_images(image_data_list, predictions_dict, num_images=num_images)

    if print_results:
        export_to_csv(sorted_image_data)

    end_time = time.time()
    duration = end_time - start_time
    console.print(
        f"Script executed successfully in {duration:.2f} seconds.", style=SUCCESS_STYLE
    )

    return sorted_image_data


def calculate_bb_area(label: Label) -> float:
    """
    Calculate the surface area of a bounding box from a Label object.

    The Label object contains class_id, center_x, center_y, width, and height.
    This function calculates the surface area of the bounding box defined by the
    width and height in the Label object.

    Args:
    label (Label): A Label object representing the bounding box and class ID.

    Returns:
    float: The surface area of the bounding box.
    """

    area = label.width * label.height

    return area


def plot_bb_distribution(label_paths: List[str], save: bool = False) -> None:
    """
    Plots the distribution of bounding box areas from a list of YOLO label file paths.

    Args:
        label_paths (List[str]): A list of paths to YOLO label files.
        save (bool): If True, saves the plot to a file named 'bbox_distribution.png' in
                     the 'pythonpix_results' directory. Defaults to False.
    """
    areas = []

    for path in label_paths:
        labels = read_yolo_labels(path)
        for label in labels:
            area = calculate_bb_area(label)
            areas.append(area)

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, color="blue", alpha=0.7)
    plt.title("Distribution of Bounding Box Areas")
    plt.xlabel("Area")
    plt.ylabel("Frequency")

    if save:
        os.makedirs("pythopix_results", exist_ok=True)
        plt.savefig("pythopix_results/bbox_distribution.png")

    plt.show()
