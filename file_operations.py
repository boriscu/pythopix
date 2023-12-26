from typing import List
from torch import Tensor
import os


def save_predictions(
    image_path: str,
    predicted_boxes: List[Tensor],
    predicted_classes: List[int],
) -> None:
    """
    Saves the predicted bounding boxes and classes to a text file in the additional_augmentation folder.

    Args:
    image_path (str): Path of the image being processed.
    predicted_boxes (list): List of predicted bounding boxes in [x1, y1, x2, y2] format.
    predicted_classes (list): List of classes corresponding to each bounding box.
    """

    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    prediction_file = os.path.join(
        "pythopix_results/additional_augmentation", base_name + ".txt"
    )

    with open(prediction_file, "w") as file:
        for cls, box in zip(predicted_classes, predicted_boxes):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]

            formatted_line = (
                f"{cls:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )
            file.write(formatted_line)
