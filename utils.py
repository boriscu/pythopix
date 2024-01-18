import os
import random
from typing import Tuple
from .data_handling import ImageData


def custom_sort_key(image_data: ImageData) -> Tuple:
    """
    Custom sorting key function for sorting ImageData objects.

    This function is used to sort ImageData instances based on false positives and box loss,
    with special handling for string values of box_loss.

    Args:
    image_data (ImageData): An instance of ImageData to be sorted.

    Returns:
    tuple: A tuple containing sorting keys.
    """

    if isinstance(image_data.box_loss, str):
        if image_data.box_loss == "Inf":
            return (
                image_data.false_positives,
                image_data.false_negatives,
                float("inf"),
            )
        elif image_data.box_loss == "No GT, FP detected":
            return (
                image_data.false_positives,
                image_data.false_negatives,
                0,
            )  # Assign a specific value for this case
    elif image_data.box_loss is None:
        return (image_data.false_positives, image_data.false_negatives, -float("inf"))
    else:
        return (
            image_data.false_positives,
            image_data.false_negatives,
            image_data.box_loss,
        )


def get_unique_folder_name(base_folder):
    """
    Generates a unique folder name by appending a number if the folder exists and is not empty.
    """
    folder = base_folder
    counter = 1
    while os.path.exists(folder) and os.listdir(folder):
        folder = f"{base_folder}_{counter}"
        counter += 1
    return folder


def get_random_files(folder: str, num_files: int) -> list:
    """Returns a list of randomly selected filenames from a folder."""
    files = os.listdir(folder)
    return random.sample(files, min(len(files), num_files))


def check_overlap_and_area(
    x1: int,
    y1: int,
    w1: int,
    h1: int,
    other: Tuple[int, int, int, int],
    threshold: int = 0.2,
) -> bool:
    """
    Check if two rectangles overlap and if the overlap is more than 20%.

    Args:
    x1 (int): X-coordinate of the top-left corner of the first rectangle.
    y1 (int): Y-coordinate of the top-left corner of the first rectangle.
    w1 (int): Width of the first rectangle.
    h1 (int): Height of the first rectangle.
    other (Tuple[int, int, int, int]): A tuple containing the x-coordinate, y-coordinate,
                                        width, and height of the second rectangle.
    threshold (int) : Threshold to determine if the overlap has happened, default set to 20%
    Returns:
    bool: True if there is an overlap and it is more than the threshold of the area of the first rectangle, False otherwise.
    """
    x2, y2, w2, h2 = other

    overlap_x1 = max(x1, x2)
    overlap_y1 = max(y1, y2)
    overlap_x2 = min(x1 + w1, x2 + w2)
    overlap_y2 = min(y1 + h1, y2 + h2)

    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        cutout_area = w1 * h1

        if overlap_area / cutout_area > threshold:
            return True

    return False
