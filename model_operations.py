import os
import shutil
from typing import List, Tuple
import torch
from torch import Tensor
from ultralytics import YOLO
from tqdm import tqdm

from .data_handling import ImageData
from .file_operations import save_predictions
from .theme import console, INFO_STYLE, SUCCESS_STYLE


def bbox_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """
    Calculate the Intersection over Union (IoU) between two sets of bounding boxes.

    The IoU is a metric used to measure the overlap between two bounding boxes. It is calculated as the area of overlap
    between the two boxes divided by the area of union of the two boxes. This function assumes that the bounding boxes
    are in the format [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
    of each box. This format implies that both the start and end pixels are inclusive in the calculation.

    Parameters:
    - box1 (Tensor): A tensor of bounding boxes, each represented by [x1, y1, x2, y2].
    - box2 (Tensor): A tensor of bounding boxes, each in the same format as box1.

    Returns:
    - Tensor: The IoU values for each corresponding pair of boxes in box1 and box2.

    The function calculates the IoU as follows:
    1. Determines the coordinates of the intersection rectangle by finding the maximum of the starting coordinates
       (x1, y1) and the minimum of the ending coordinates (x2, y2) for each pair of bounding boxes.
    2. Calculates the area of the intersection rectangle. If the boxes do not overlap, this area is zero.
    3. Computes the area of each bounding box.
    4. Calculates the union area of the boxes as the sum of their individual areas minus the intersection area.
    5. Computes the IoU by dividing the intersection area by the union area.

    The result is a value between 0 and 1, where 1 indicates a perfect overlap and 0 indicates no overlap.
    """

    # Calculate intersection
    inter_rect_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_rect_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_rect_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_rect_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Calculate union
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area
    return iou


def segregate_images(
    image_data_list: List[ImageData], predictions_dict: dict, num_images: int = 10
) -> None:
    """
    Copies images into two separate folders based on whether they require additional augmentation.
    For images that require additional augmentation, saves their predicted labels.
    For other images, copies their original labels.

    Args:
    image_data_list (List[ImageData]): List of ImageData objects sorted based on certain criteria.
    predictions_dict (dict): Dictionary containing the predicted boxes and classes for each image.
    num_images (int): Number of top images to copy to the additional augmentation folder.
    """

    parent_folder = "pythopix_results"

    additional_augmentation_folder = os.path.join(
        parent_folder, "additional_augmentation"
    )
    count = 1
    while os.path.exists(additional_augmentation_folder):
        additional_augmentation_folder = f"{additional_augmentation_folder}_{count}"
        count += 1

    no_additional_augmentation_folder = os.path.join(
        parent_folder, "no_additional_augmentation"
    )
    count = 1
    while os.path.exists(no_additional_augmentation_folder):
        no_additional_augmentation_folder = (
            f"{no_additional_augmentation_folder}_{count}"
        )
        count += 1

    # Create parent directory if it doesn't exist
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Create subdirectories
    os.makedirs(additional_augmentation_folder, exist_ok=True)
    os.makedirs(no_additional_augmentation_folder, exist_ok=True)

    console.print(
        "Segregating data...",
        style=INFO_STYLE,
    )
    for i, image_data in tqdm(enumerate(image_data_list), desc="Processing images"):
        image_basename = os.path.basename(image_data.image_path)
        label_path = image_data.image_path.rsplit(".", 1)[0] + ".txt"

        # For images that require additional augmentation
        if i < num_images:
            shutil.copy(
                image_data.image_path,
                os.path.join(additional_augmentation_folder, image_basename),
            )
            if image_data.image_path in predictions_dict:
                predictions = predictions_dict[image_data.image_path]
                save_predictions(
                    os.path.join(additional_augmentation_folder, image_basename),
                    predictions["predicted_boxes"],
                    predictions["predicted_classes"],
                )
        # For other images
        else:
            shutil.copy(
                image_data.image_path,
                os.path.join(no_additional_augmentation_folder, image_basename),
            )
            if os.path.exists(label_path):
                shutil.copy(
                    label_path,
                    os.path.join(
                        no_additional_augmentation_folder, os.path.basename(label_path)
                    ),
                )

    for image_data in image_data_list[num_images:]:
        shutil.copy(
            image_data.image_path,
            os.path.join(
                no_additional_augmentation_folder,
                os.path.basename(image_data.image_path),
            ),
        )

        label_path = image_data.image_path.rsplit(".", 1)[0] + ".txt"
        if os.path.exists(label_path):
            shutil.copy(
                label_path,
                os.path.join(
                    no_additional_augmentation_folder, os.path.basename(label_path)
                ),
            )

    console.print(
        "Segragation successful",
        style=SUCCESS_STYLE,
    )


def process_image(
    image_path: str,
    model: YOLO = None,
    model_path: str = None,
    verbose: bool = False,
) -> Tuple[ImageData, dict]:
    """
    Processes an image using a YOLO model and calculates metrics such as false positives,
    false negatives, and box loss. The function allows the use of a custom model or loads a model from a specified path.

    Args:
    image_path (str): The file path of the image to be processed.
    model (YOLO, optional): An instance of the YOLO model used for object detection. If None, the model is loaded from `model_path`.
    model_path (str, optional): The file path to load the YOLO model from. Used if `model` is None.
    verbose (bool, optional): If True, provides detailed output during the prediction process. Default is False.

    Returns:
    Tuple[ImageData, dict]:
        - ImageData: An object containing detailed information about the image, including metrics like false positives, false negatives, and box loss.
        - dict: A dictionary containing the predicted bounding boxes and the classes of the detected objects.
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

    result = model(image_path, verbose=verbose)[0]
    predicted_classes = result.boxes.cls.tolist()
    predicted_boxes = result.boxes.xyxyn

    # Get the ground-truth classes
    gt_classes = []
    gt_labels = []
    label_path = os.path.join(
        os.path.dirname(image_path),
        os.path.basename(image_path).rsplit(".", 1)[0] + ".txt",
    )

    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            for line in file:
                gt_label = line.strip().split(" ")
                gt_class = float(gt_label[0])
                gt_classes.append(gt_class)
                gt_x, gt_y, gt_w, gt_h = map(float, gt_label[1:])
                gt_labels.append([gt_x, gt_y, gt_w, gt_h])

    gt_boxes = torch.tensor(gt_labels)

    # Calculate the false positives
    false_positives = sum(
        1 for pred_class in predicted_classes if pred_class not in gt_classes
    )

    false_negatives = sum(
        1 for gt_class in gt_classes if gt_class not in predicted_classes
    )

    if gt_boxes.nelement() != 0 and predicted_boxes.nelement() != 0:
        # The 1 at the end of the torch.cat function is specifying that the concatenation should be done along the columns (dimension 1),
        # resulting in a tensor where each bounding box is represented as [x1, y1, x2, y2].

        gt_boxes_wh = torch.cat(
            (
                gt_boxes[:, :2] - gt_boxes[:, 2:] / 2,
                gt_boxes[:, :2] + gt_boxes[:, 2:] / 2,
            ),
            1,
        )
        predicted_boxes_wh = torch.cat(
            (
                predicted_boxes[:, :2] - predicted_boxes[:, 2:] / 2,
                predicted_boxes[:, :2] + predicted_boxes[:, 2:] / 2,
            ),
            1,
        )

        # Calculate IoU for each predicted box against all ground truth boxes
        ious = torch.stack(
            [
                bbox_iou(pred_box.unsqueeze(0), gt_boxes_wh)
                for pred_box in predicted_boxes_wh
            ]
        )

        # Average IoU for this image
        avg_iou = ious.mean()
        box_loss = 1 - avg_iou.item()  # Box loss is 1 - IoU
    elif gt_boxes.nelement() == 0 and predicted_boxes.nelement() != 0:
        box_loss = "No GT, FP detected"
    elif gt_boxes.nelement() != 0 and predicted_boxes.nelement() == 0:
        box_loss = "Inf"
    else:
        box_loss = None  # No ground truth or no prediction

    return ImageData(image_path, false_positives, false_negatives, box_loss), {
        "predicted_boxes": predicted_boxes,
        "predicted_classes": predicted_classes,
    }
