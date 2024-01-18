import os
import random
from .data_handling import ImageData


def custom_sort_key(image_data: ImageData) -> tuple:
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
