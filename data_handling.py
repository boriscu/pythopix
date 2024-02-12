import csv
import tqdm
from typing import List, Tuple
from .theme import console, INFO_STYLE
import cv2
import os
import numpy as np


class ImageData:
    """
    A class to represent image data for analysis.

    Attributes:
    image_path (str): Path of the image.
    false_positives (int): Count of false positive detections.
    false_negatives (int): Count of false negative detections.
    box_loss (str or float): Loss calculated based on IoU, or a string indicating special conditions.

    Methods:
    __repr__: Returns a formatted string representation of the image data.
    """

    def __init__(
        self,
        image_path: str,
        false_positives: int,
        false_negatives: int,
        box_loss: float,
    ):
        self.image_path = image_path
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.box_loss = box_loss

    def __repr__(self) -> str:
        return f"ImagePath: {self.image_path}, FalsePositives: {self.false_positives}, FalseNegatives: {self.false_negatives}, BoxLoss: {self.box_loss}"


def export_to_csv(
    image_data_list: List[ImageData], filename: str = "image_data_results.csv"
) -> None:
    """
    Exports the image data results to a CSV file.

    Args:
    image_data_list (List[ImageData]): List of ImageData objects to export.
    filename (str): Name of the CSV file to create.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["ImagePath", "FalsePositives", "FalseNegatives", "BoxLoss"])
        console.print(
            "Writing data to csv...",
            style=INFO_STYLE,
        )
        for image_data in tqdm.tqdm(image_data_list):
            writer.writerow(
                [
                    image_data.image_path,
                    image_data.false_positives,
                    image_data.false_negatives,
                    image_data.box_loss,
                ]
            )


def resize_images_in_folder(
    folder_path: str,
    target_width: int,
    target_height: int,
    width_deviation: int,
    height_deviation: int,
) -> None:
    """
    Resizes all PNG images in the specified folder to random sizes based on target dimensions and deviations.

    The function randomly selects a new width and height for each image within the specified deviation from the target
    dimensions. This approach helps introduce variability in the dataset while roughly maintaining the average
    dimensions.

    Parameters:
    - folder_path (str): Path to the folder containing the images to be resized.
    - target_width (int): Target width for the resizing operation.
    - target_height (int): Target height for the resizing operation.
    - width_deviation (int): Allowed deviation from the target width to introduce randomness.
    - height_deviation (int): Allowed deviation from the target height to introduce randomness.

    Returns:
    - None: The function modifies the images in place and does not return a value.
    """
    for filename in tqdm.tqdm(os.listdir(folder_path), desc="Resizing images"):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            new_width = np.random.randint(
                target_width - width_deviation, target_width + width_deviation + 1
            )
            new_height = np.random.randint(
                target_height - height_deviation, target_height + height_deviation + 1
            )

            resized_image = cv2.resize(image, (new_width, new_height))

            cv2.imwrite(image_path, resized_image)
