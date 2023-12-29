import csv
import os
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from typing import Optional, List, Dict, Tuple
import numpy as np

from .data_handling import export_to_csv
from .model_operations import process_image, segregate_images
from .utils import custom_sort_key
from .theme import console, INFO_STYLE, SUCCESS_STYLE
from .labels_operations import Label, extract_label_files, read_yolo_labels


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

    for image_path in tqdm(images, desc="Processing Images"):
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
    float: The fractional surface area of the bounding box, as a proportion of the total image area.
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
            area = calculate_bb_area(label) * 100
            areas.append(area)

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, color="blue", alpha=0.7)
    plt.title("Distribution of Bounding Box Areas")
    plt.xlabel("Area (% of original image)")
    plt.ylabel("Frequency")

    if save:
        os.makedirs("pythopix_results", exist_ok=True)
        plt.savefig("pythopix_results/bbox_distribution.png")

    plt.show()


def calculate_segmented_metrics(
    folder_path: str,
    model: YOLO = None,
    model_path: str = None,
    segment_number: int = 4,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Processes a folder of images, dividing bounding boxes into segments based on their sizes,
    and calculates average metrics for each segment.

    Args:
    folder_path (str): Path to the folder containing images and corresponding YOLO label files.
    model (YOLO, optional): An instance of the YOLO model. If None, model is loaded from model_path.
    model_path (str, optional): Path to load the YOLO model, used if model is None.
    segment_number (int, optional): Number of segments to divide the bounding boxes into based on their sizes.

    Returns:
    Dict[str, Tuple[float, float, float]]: A dictionary where the key is the segment range (e.g., '0-0.25'),
    and the value is a tuple containing the average false positives, false negatives, and box loss for that segment.
    """

    if model is None:
        if model_path is not None:
            model = YOLO(model_path)
        else:
            console.print(
                "Model path not provided or not found. Using default YOLO model.",
                style=INFO_STYLE,
            )
            model = YOLO("yolov8n")

    # Extract label files
    label_files = extract_label_files(folder_path, label_type="txt")

    # Calculate Bounding Box Sizes
    all_bb_areas = []
    for label_file in label_files:
        labels = read_yolo_labels(label_file)
        for label in labels:
            area = calculate_bb_area(label)
            all_bb_areas.append(area)

    # Segment Bounding Boxes
    max_area = max(all_bb_areas)
    min_area = min(all_bb_areas)
    segment_size = (max_area - min_area) / segment_number
    segments = [
        (
            round(min_area + i * segment_size, 2),
            round(min_area + (i + 1) * segment_size, 2),
        )
        for i in range(segment_number)
    ]
    # Initialize metrics storage
    metrics_by_segment = {f"{seg[0]:.2f}-{seg[1]:.2f}": [] for seg in segments}

    # Assign Bounding Boxes to Segments and Calculate Metrics
    for label_file in tqdm(label_files, desc="Calculating metrics"):
        image_file = label_file.replace(".txt", ".png")
        labels = read_yolo_labels(label_file)

        for label in labels:
            area = calculate_bb_area(label)
            for seg in segments:
                if seg[0] <= area < seg[1]:
                    segment_key = f"{seg[0]:.2f}-{seg[1]:.2f}"
                    image_data, _ = process_image(image_file, model, verbose=False)
                    metrics_by_segment[segment_key].append(image_data)

    for segment, data in tqdm(
        metrics_by_segment.items(), desc="Calculating average metrics by segment"
    ):
        if data:
            valid_fp = [
                d.false_positives
                for d in data
                if isinstance(d.false_positives, (int, float))
            ]
            valid_fn = [
                d.false_negatives
                for d in data
                if isinstance(d.false_negatives, (int, float))
            ]
            valid_bl = [
                d.box_loss for d in data if isinstance(d.box_loss, (int, float))
            ]

            avg_false_positives = round(np.mean(valid_fp) if valid_fp else 0, 2)
            avg_false_negatives = round(np.mean(valid_fn) if valid_fn else 0, 2)
            avg_box_loss = round(np.mean(valid_bl) if valid_bl else 0, 2)

            metrics_by_segment[segment] = (
                avg_false_positives,
                avg_false_negatives,
                avg_box_loss,
            )
        else:
            metrics_by_segment[segment] = (0, 0, 0)

    return metrics_by_segment


def plot_metrics_by_segment(
    metrics_by_segment: Dict[str, Tuple[float, float, float]], save: bool = False
) -> None:
    """
    Plots and optionally saves three bar charts for the given metrics by segment.
    There will be one chart for false positives, one for false negatives, and one for box loss.

    Args:
    metrics_by_segment (Dict[str, Tuple[float, float, float]]): A dictionary with segment ranges as keys and tuples of metrics as values.
    save (bool, optional): If True, saves the plots to the 'pythopix_results' folder. Defaults to False.
    """
    segments = [
        f"{float(seg.split('-')[0])*100:.2f}-{float(seg.split('-')[1])*100:.2f}%"
        for seg in metrics_by_segment.keys()
    ]
    false_positives = [metrics[0] for metrics in metrics_by_segment.values()]
    false_negatives = [metrics[1] for metrics in metrics_by_segment.values()]
    box_losses = [metrics[2] for metrics in metrics_by_segment.values()]

    # Create the directory for saving results if it doesn't exist
    if save:
        os.makedirs("pythopix_results", exist_ok=True)

    # Plotting False Positives
    plt.figure(figsize=(10, 6))
    plt.bar(segments, false_positives, color="#56baf0")
    plt.xlabel("Segments (% of original image)")
    plt.ylabel("False Positives")
    plt.title("False Positives by Segment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig("pythopix_results/false_positives_by_segment.png")
    plt.show()

    # Plotting False Negatives
    plt.figure(figsize=(10, 6))
    plt.bar(segments, false_negatives, color="#a3e38c")
    plt.xlabel("Segments (% of original image)")
    plt.ylabel("False Negatives")
    plt.title("False Negatives by Segment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig("pythopix_results/false_negatives_by_segment.png")
    plt.show()

    # Plotting Box Loss
    plt.figure(figsize=(10, 6))
    plt.bar(segments, box_losses, color="#050c26")
    plt.xlabel("Segments (% of original image)")
    plt.ylabel("Box Loss")
    plt.title("Box Loss by Segment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig("pythopix_results/box_loss_by_segment.png")
    plt.show()


def save_segmented_metrics_to_csv(
    metrics_by_segment: Dict[str, Tuple[float, float, float]]
) -> None:
    """
    Saves the metrics by segment data to a CSV file.

    Args:
    metrics_by_segment (Dict[str, Tuple[float, float, float]]): Metrics data segmented by bounding box sizes.
    save (bool, optional): If True, saves the data to a CSV file in the 'pythopix_results' folder. Defaults to False.
    """
    os.makedirs("pythopix_results", exist_ok=True)
    file_path = os.path.join("pythopix_results", "metrics_by_segment.csv")

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Segments", *metrics_by_segment.keys()])

        false_positives = ["False Positives"] + [
            metrics[0] for metrics in metrics_by_segment.values()
        ]
        false_negatives = ["False Negatives"] + [
            metrics[1] for metrics in metrics_by_segment.values()
        ]
        box_loss = ["Box Loss"] + [
            metrics[2] for metrics in metrics_by_segment.values()
        ]

        # Writing data rows
        writer.writerow(false_positives)
        writer.writerow(false_negatives)
        writer.writerow(box_loss)
