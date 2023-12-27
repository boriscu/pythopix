from typing import Optional, Tuple, List
import cv2
import matplotlib.pyplot as plt
import os
import time
import glob
from tqdm import tqdm
from .theme import ERROR_STYLE, INFO_STYLE, SUCCESS_STYLE, console


def yolo_to_bbox(
    yolo_data: List[float], img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Converts YOLO format bounding box data to pixel coordinates.

    This function takes bounding box data in YOLO format (center x, center y, width, height)
    and converts it to pixel coordinates (xmin, ymin, xmax, ymax) based on the dimensions
    of the image.

    Parameters:
    yolo_data (List[float]): A list containing four float values representing the bounding box
                             in YOLO format: [x_center, y_center, width, height].
    img_width (int): The width of the image.
    img_height (int): The height of the image.

    Returns:
    Tuple[int, int, int, int]: A tuple containing four integer values representing the bounding box
                               in pixel coordinates: (xmin, ymin, xmax, ymax).
    """
    x_center, y_center, width, height = yolo_data
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return x_min, y_min, x_max, y_max


def add_bboxes_to_image(
    image: List[List[List[int]]], label_file: str, img_width: int, img_height: int
) -> List[List[List[int]]]:
    """
    Adds bounding boxes to an image based on label data.

    This function reads a label file with bounding box data in YOLO format and draws these
    bounding boxes on the provided image. If the label file does not exist, the original image
    is returned without modification.

    Parameters:
    image (List[List[List[int]]]): The image data in OpenCV format (BGR).
    label_file (str): Path to the label file containing bounding boxes in YOLO format.
    img_width (int): The width of the image.
    img_height (int): The height of the image.

    Returns:
    List[List[List[int]]]: The image data in OpenCV format (BGR).
    """
    if not os.path.exists(label_file):
        console.print("Label file not found. Box will not be drawn.", style=ERROR_STYLE)
        return image

    with open(label_file, "r") as file:
        for line in file:
            bbox_data = [
                float(x) for x in line.strip().split()[1:]
            ]  # Skip the class label
            bbox = yolo_to_bbox(bbox_data, img_width, img_height)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    return image


def compare_labels(
    image_path: str,
    predicted_label_path: str,
    original_label_path: Optional[str] = None,
    show: bool = True,
    save_fig: bool = False,
) -> None:
    """
    Compares and displays bounding boxes from original and predicted label files on an image.

    This function reads an image and its corresponding label files (original and predicted),
    draws bounding boxes on the image as per the labels, and can display and/or save the
    comparison as a side-by-side plot.

    Parameters:
    image_path (str): Path to the original image file.
    predicted_label_path (str): Path to the predicted label file in YOLO format.
    original_label_path (Optional[str]): Path to the original label file in YOLO format. If not provided,
                                         the function will try to find a label file with the same name as
                                         the image in its directory. Defaults to None.
    show (bool): If True, the function will display the comparison plot. Defaults to True.
    save_fig (bool): If True, the function will save the comparison plot in the directory
                 'pythopix_results/figs' with the filename format '{image_name}-compared.png'.
                 Defaults to False.

    Returns:
    None: The function does not return anything but may show and/or save an image based on the arguments.
    """
    if original_label_path is None:
        base_name = os.path.basename(image_path)
        name_without_ext, _ = os.path.splitext(base_name)
        original_label_path = os.path.join(
            os.path.dirname(image_path), name_without_ext + ".txt"
        )

    img = cv2.imread(image_path)
    if img is None:
        console.print(
            f"Image {image_path} not found, skipping.",
            style=ERROR_STYLE,
        )
        return

    img_height, img_width = img.shape[:2]
    original_img = add_bboxes_to_image(
        img.copy(), original_label_path, img_width, img_height
    )
    predicted_img = add_bboxes_to_image(
        img.copy(), predicted_label_path, img_width, img_height
    )

    if show or save_fig:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Predicted")
        axs[1].axis("off")

        if save_fig:
            save_folder = "pythopix_results/figs"
            os.makedirs(save_folder, exist_ok=True)
            output_filename = os.path.join(
                save_folder, f"{name_without_ext}-compared.png"
            )
            plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()


def compare_folder_labels(
    image_folder: str,
    labels: List[str],
    limit: Optional[int] = None,
    show: bool = False,
    save: bool = True,
) -> None:
    """
    Compares original image labels with their corresponding predicted labels in another folder.

    This function processes each image and its original label in the specified image folder, compares it with the
    corresponding label in the label folder, and optionally displays or saves the comparison.

    Parameters:
    image_folder (str): Path to the folder containing images.
    labels (List[str]): Predicted labels.
    limit (Optional[int]): The maximum number of images to process. If None, all images in the folder are processed. Defaults to None.
    show (bool): If True, displays the comparison plot for each image. Defaults to False.
    save (bool): If True, saves the comparison plot for each image. Defaults to True.

    Returns:
    None: The function does not return anything but may show and/or save images based on the arguments.

    """
    start_time = time.time()
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))

    label_map = {
        os.path.splitext(os.path.basename(label))[0]: label for label in labels
    }

    if limit is not None:
        image_paths = image_paths[:limit]

    save_folder = "pythopix_results/figs"
    os.makedirs(save_folder, exist_ok=True)

    for image_path in tqdm(
        image_paths,
        desc="Processing images",
    ):
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        label = label_map.get(base_name)
        original_label_file = os.path.join(image_folder, base_name + ".txt")

        if label is None:
            console.print(
                f"No label file found for image {image_path}, skipping.",
                style=ERROR_STYLE,
            )
            continue

        original_img = cv2.imread(image_path)
        if original_img is None:
            console.print(f"Image {image_path} not found, skipping.", style=INFO_STYLE)
            continue
        predicted_img = original_img.copy()

        if show or save:
            img_height, img_width = original_img.shape[:2]

            original_img = add_bboxes_to_image(
                original_img, original_label_file, img_width, img_height
            )
            predicted_img = add_bboxes_to_image(
                predicted_img, label, img_width, img_height
            )

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Predicted")
            axs[1].axis("off")

            if save:
                save_path = os.path.join(save_folder, f"{base_name}_compared.png")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

            if show:
                plt.show()

    end_time = time.time()
    console.print(
        f"Success. Total time taken: {end_time - start_time:.2f} seconds",
        style=SUCCESS_STYLE,
    )
