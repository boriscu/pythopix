import os
import time
import torch
import tqdm

from ultralytics import YOLO
from .data_handling import export_to_csv
from .model_operations import process_image, segregate_images
from .utils import custom_sort_key
from typing import Optional, List
from .theme import console, INFO_STYLE, SUCCESS_STYLE


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
