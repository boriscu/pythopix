import csv
import tqdm
from typing import List
from .theme import console, INFO_STYLE


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
