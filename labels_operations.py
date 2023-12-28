import os
from typing import List, Dict, NamedTuple
import shutil
import json
from tqdm import tqdm
import numpy as np
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
