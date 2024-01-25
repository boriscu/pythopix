import os
import time
from typing import List, Dict, NamedTuple, Optional, Tuple
import shutil
import json
import cv2
from tqdm import tqdm
import numpy as np

from .utils import check_overlap_and_area
from .theme import ERROR_STYLE, console, INFO_STYLE, SUCCESS_STYLE


class Label(NamedTuple):
    """
    Represents a label in YOLO format.

    Attributes:
        class_id (int): The class ID of the object in the bounding box.
        x_center (float): The x-coordinate of the center of the bounding box,
                          normalized to the image width.
        y_center (float): The y-coordinate of the center of the bounding box,
                          normalized to the image height.
        width (float): The width of the bounding box, normalized to the image width.
        height (float): The height of the bounding box, normalized to the image height.

    The coordinates and dimensions are normalized relative to the image dimensions,
    meaning they are expressed as a fraction of the image's width and height.
    For instance, an x_center of 0.5 would mean the center of the box is at the
    middle of the image width.
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def read_yolo_labels(file_path: str) -> List[Label]:
    """
    Reads a YOLO label file and returns a list of labels.

    Each line in the YOLO label file should have the format:
    "class_id center_x center_y width height"

    Args:
    file_path (str): Path to the YOLO label file.

    Returns:
    List[Label]: A list of Label objects parsed from the file.
    """
    labels = []

    if not os.path.exists(file_path):
        console.print(
            "Can't parse labels from file, no file named: {file_path} found",
            style=ERROR_STYLE,
        )
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            labels.append(Label(int(class_id), x_center, y_center, width, height))

    return labels


def extract_label_sizes(label_files: List[str]) -> Tuple[List[int], List[int]]:
    """
    Extracts the widths and heights of bounding boxes in pixels from YOLO label files.

    This function iterates through a list of YOLO label files, reading the normalized
    bounding box dimensions (width and height) for each box in each file. It then
    converts these normalized dimensions to pixel dimensions based on the corresponding
    image size.

    Note:
    This function assumes that each label file has a corresponding image file in the
    same directory and with the same base filename but different extension (.jpg).

    Parameters:
    - label_files (List[str]): A list of file paths to YOLO label files.

    Returns:
    - Tuple[List[int], List[int]]: Two lists containing the widths and heights of the
                                   bounding boxes in pixels, respectively.
    """
    widths, heights = [], []
    for label_file in tqdm(label_files, desc="Reading label sizes"):
        image_file = label_file.replace(".txt", ".png")
        if os.path.exists(image_file):
            image = cv2.imread(image_file)
            img_height, img_width = image.shape[:2]

            with open(label_file, "r") as file:
                for line in file:
                    _, x_center, y_center, width, height = map(float, line.split())
                    pixel_width = int(width * img_width)
                    pixel_height = int(height * img_height)
                    widths.append(pixel_width)
                    heights.append(pixel_height)
        else:
            print(f"Image file corresponding to {label_file} not found.")

    return widths, heights


def extract_label_files(source_folder: str, label_type: str = "txt") -> List[str]:
    """
    Extracts the full paths of all label files (`.txt` or `.json`) from the specified source folder.

    Args:
        source_folder (str): The path to the folder from which to extract label file paths.
        label_type (str): The type of label files to extract ('txt' or 'json'). Defaults to 'txt'.

    Returns:
        List[str]: A list containing the full paths of label files found in the source folder.

    """
    if not os.path.exists(source_folder):
        console.print(
            "Error: The source folder for label extraction does not exist.",
            style=ERROR_STYLE,
        )
        raise Exception("The source folder does not exist.")

    file_extension = ".txt" if label_type == "txt" else ".json"
    label_files = [
        os.path.join(source_folder, filename)
        for filename in tqdm(os.listdir(source_folder), desc="Extracting label files")
        if filename.endswith(file_extension)
    ]
    console.print("Successfuly extracted labels", style=SUCCESS_STYLE)
    return label_files


def save_extracted_labels(
    label_files: List[str], destination_folder: str = "pythopix_results/txt_labels"
):
    """
    Saves the `.txt` files specified in the label_files list to the destination folder.

    Args:
        label_files (List[str]): A list of full paths to `.txt` file names to be saved.
        destination_folder (str): The path to the folder where `.txt` files will be saved. Defaults to 'pythopix_results/saved_labels'.

    Note:
        The function creates the destination folder if it does not exist.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_path in tqdm(label_files):
        shutil.copy(
            file_path, os.path.join(destination_folder, os.path.basename(file_path))
        )


def convert_txt_to_json_labels(
    txt_label_files: List[str],
    label_mapping: Dict[int, str],
    destination_folder: str = "pythopix_results/json_labels",
    image_height: int = 1080,
    image_width: int = 1920,
):
    """
    Converts `.txt` label files to `.json` format and saves them in the specified destination folder.

    Args:
        txt_label_files (List[str]): A list of full paths to `.txt` label files.
        label_mapping (Dict[int, str]): A dictionary mapping numeric labels to string labels.
        destination_folder (str): The path to the folder where `.json` label files will be saved. Defaults to 'pythopix_results/json_labels'.
        image_height (int): The height of the images associated with the labels. Defaults to 1080.
        image_width (int): The width of the images associated with the labels. Defaults to 1920.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    console.print(
        "Converting txt to json labels...",
        style=INFO_STYLE,
    )
    for txt_file_path in tqdm(txt_label_files):
        base_name = os.path.basename(txt_file_path).replace(".txt", "")
        json_filename = f"{base_name}.json"
        path_json = os.path.join(destination_folder, json_filename)

        with open(txt_file_path, "r") as file:
            lines = file.read().split("\n")
        shapes = []
        for line in lines:
            if line:
                target, roi_center_x, roi_center_y, roi_width, roi_height = np.array(
                    line.split(" ")
                ).astype("float")[0:5]
                roi_w2, roi_h2 = roi_width / 2, roi_height / 2

                label = label_mapping.get(int(target), "Unknown")
                shape = {
                    "label": label,
                    "points": [
                        [
                            (roi_center_x - roi_w2) * image_width,
                            (roi_center_y - roi_h2) * image_height,
                        ],
                        [
                            (roi_center_x + roi_w2) * image_width,
                            (roi_center_y + roi_h2) * image_height,
                        ],
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
                shapes.append(shape)

        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{base_name}.png",
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }

        with open(path_json, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    console.print(
        "Conversion successful.",
        style=SUCCESS_STYLE,
    )


def convert_json_to_txt_labels(
    json_label_files: List[str],
    base_destination_folder: str = "pythopix_results/txt_labels",
):
    """
    Converts `.json` label files to `.txt` format and saves them in a specified destination folder.

    Args:
        json_label_files (List[str]): A list of full paths to `.json` label files.
        base_destination_folder (str): The base path for the folder where `.txt` label files will be saved.
                                       Defaults to 'pythopix_results/txt_labels'.
    """
    destination_folder = base_destination_folder
    count = 1
    while os.path.exists(destination_folder):
        destination_folder = f"{base_destination_folder}_{count}"
        count += 1

    os.makedirs(destination_folder)

    console.print(
        "Converting json to txt labels...",
        style=INFO_STYLE,
    )

    for json_file_path in tqdm(json_label_files):
        with open(json_file_path, "r") as file:
            data = json.load(file)

        img_w, img_h = data["imageWidth"], data["imageHeight"]
        labels = []

        for shape in data["shapes"]:
            label_data = shape["label"].lower()
            if label_data == "bush" or label_data == "tree":
                label_out = 0.0
            elif label_data == "pole":
                label_out = 1.0
            else:
                print(f"Invalid label at img: {json_file_path}")
                continue

            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]

            roi_width = np.abs(x2 - x1) / img_w
            roi_height = np.abs(y2 - y1) / img_h
            roi_center_x = np.mean([x1, x2]) / img_w
            roi_center_y = np.mean([y1, y2]) / img_h

            labels.append(
                [label_out, roi_center_x, roi_center_y, roi_width, roi_height]
            )

        txt_filename = os.path.basename(json_file_path).replace(".json", ".txt")
        path_out = os.path.join(destination_folder, txt_filename)
        np.savetxt(path_out, labels, fmt="%f")

    console.print(
        "Conversion successful.",
        style=SUCCESS_STYLE,
    )


def filter_and_resize_labels(
    input_folder: str,
    output_folder: str = "pythopix_results/filtered_labels",
    min_width: int = 50,
    min_height: int = 50,
    resize_aspect_ratio: Tuple[int, int] = (100, 100),
    alpha: float = 0.2,
    allowed_classes: Optional[List[int]] = None,
) -> None:
    """
    Processes images from the input folder, filters labels based on non-overlapping and
    minimum size criteria, then resizes and saves them to the output folder.

    Args:
    - input_folder (str): Path to the folder containing images and their corresponding YOLO label files.
    - output_folder (str): Path to the folder where the processed labels will be saved.
    - min_width (int): Minimum width in pixels for the labels to be considered.
    - min_height (int): Minimum height in pixels for the labels to be considered.
    - resize_aspect_ratio (Tuple[int, int]): The aspect ratio to resize the labels to.
    - alpha (float): Threshold for maximum allowed overlap as a fraction of the first rectangle's area.

    Returns:
    - None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()

    for image_file in tqdm(os.listdir(input_folder), desc="Filtering labels"):
        if image_file.endswith(".png"):
            image_path = os.path.join(input_folder, image_file)
            label_path = image_path.replace(".png", ".txt")

            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                img_height, img_width = image.shape[:2]
                labels = read_labels(label_path)

                filtered_labels = []
                for i, (class_id, label) in enumerate(labels):
                    if allowed_classes is None or class_id in allowed_classes:
                        pixel_label = convert_to_pixels(label, img_width, img_height)
                        if pixel_label[2] >= min_width and pixel_label[3] >= min_height:
                            overlap = False
                            for j, (_, other_label) in enumerate(labels):
                                if i != j and check_overlap_and_area(
                                    *label, other_label, threshold=alpha
                                ):
                                    overlap = True
                                    break
                            if not overlap:
                                filtered_labels.append(pixel_label)

                for i, label in enumerate(filtered_labels):
                    cropped_image = crop_and_resize(image, label, resize_aspect_ratio)
                    cv2.imwrite(
                        os.path.join(
                            output_folder,
                            f"{os.path.splitext(image_file)[0]}_label_{i}.png",
                        ),
                        cropped_image,
                    )
    end_time = time.time()
    console.print(
        f"Successfuly filtered labels, took {round(end_time-start_time,2)} seconds",
        style=SUCCESS_STYLE,
    )


def read_labels(label_file: str) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """
    Reads YOLO label files and returns a list of labels with class IDs and normalized bounding box coordinates.

    Each label in the label file is expected to be in the format:
    class_id x_center y_center width height, where all values are normalized relative to the image size.

    Args:
    - label_file (str): Path to the YOLO label file.

    Returns:
    - List[Tuple[int, Tuple[float, float, float, float]]]: A list of tuples where each tuple contains:
        - int: The class ID of the label.
        - Tuple[float, float, float, float]: The normalized bounding box coordinates (x_center, y_center, width, height).
    """
    labels = []
    with open(label_file, "r") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            labels.append((int(class_id), (x_center, y_center, width, height)))
    return labels


def convert_to_pixels(
    label: Tuple[float, float, float, float], img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Converts normalized YOLO label coordinates to pixel coordinates based on the image dimensions.

    Args:
    - label (Tuple[float, float, float, float]): Normalized bounding box coordinates (x_center, y_center, width, height).
    - img_width (int): The width of the image in pixels.
    - img_height (int): The height of the image in pixels.

    Returns:
    - Tuple[int, int, int, int]: The bounding box coordinates in pixel values (x1, y1, width, height).
    """
    x_center, y_center, width, height = label
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    w = int(width * img_width)
    h = int(height * img_height)
    return x1, y1, w, h


def crop_and_resize(
    image: np.ndarray,
    label: Tuple[int, int, int, int],
    resize_aspect_ratio: Tuple[int, int],
) -> np.ndarray:
    x, y, w, h = label
    cropped = image[y : y + h, x : x + w]
    resized = cv2.resize(cropped, resize_aspect_ratio)
    return resized
