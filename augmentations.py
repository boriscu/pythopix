from typing import Tuple
import random
import cv2
import numpy as np
import os
import glob
import shutil
import tqdm
import time
from .theme import console, SUCCESS_STYLE, ERROR_STYLE
from .utils import get_unique_folder_name, get_random_files, check_overlap_and_area


def gaussian_noise(
    image_path: str,
    sigma_range: tuple = (30, 70),
    frequency: float = 1.0,
    noise_probability: float = 0.5,
) -> np.ndarray:
    """
    Adds Gaussian noise to an image with a certain probability and varying intensity.

    Parameters:
    image_path (str): The file path to the input image.
    sigma_range (tuple): The range of standard deviation for the Gaussian noise.
                         Noise intensity will be randomly selected within this range.
    frequency (float): The frequency of applying the noise. A value of 1.0 applies noise to every pixel,
                       while lower values apply it more sparsely.
    noise_probability (float): Probability of applying noise to the image.
                                Ranges from 0 (no noise) to 1 (always add noise).

    Returns:
    np.ndarray: The image with or without Gaussian noise added.

    Raises:
    FileNotFoundError: If the image at the specified path is not found.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    if random.random() < noise_probability:
        h, w, c = image.shape
        mean = 0

        sigma = random.uniform(*sigma_range)

        # Generate Gaussian noise
        gauss = np.random.normal(mean, sigma, (h, w, c)) * frequency
        gauss = gauss.reshape(h, w, c)

        # Add the Gaussian noise to the image
        noisy_image = image + gauss

        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)

        return noisy_image
    else:
        return image


def random_erasing(
    image_path: str,
    erasing_prob: float = 0.5,
    area_ratio_range: Tuple[float, float] = (0.02, 0.1),
    aspect_ratio_range: Tuple[float, float] = (0.3, 3),
) -> np.ndarray:
    """
    Applies the Random Erasing augmentation to an image.

    Parameters:
    image_path (str): Path to the input image.
    erasing_prob (float): Probability of erasing a random patch. Defaults to 0.5.
    area_ratio_range (Tuple[float, float]): Range of the ratio of the erased area to the whole image area. Defaults to (0.02, 0.4).
    aspect_ratio_range (Tuple[float, float]): Range of the aspect ratio of the erased area. Defaults to (0.3, 3).

    Returns:
    np.ndarray: Image with a random patch erased.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    if np.random.rand() > erasing_prob:
        return image  # Skip erasing with a certain probability

    h, w, _ = image.shape
    area = h * w

    for _ in range(100):  # Try 100 times
        erase_area = np.random.uniform(area_ratio_range[0], area_ratio_range[1]) * area
        aspect_ratio = np.random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])

        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))

        if erase_h < h and erase_w < w:
            x = np.random.randint(0, w - erase_w)
            y = np.random.randint(0, h - erase_h)
            image[y : y + erase_h, x : x + erase_w] = 0
            return image

    return image


# Available augmentation functions
augmentation_funcs = {"gaussian": gaussian_noise, "random_erase": random_erasing}


def apply_augmentations(
    input_folder: str, augmentation_type: str, output_folder: str = None, **kwargs
):
    """
    Applies a specified type of augmentation to all images in a given folder and saves the results along with their
    corresponding label files to an output folder. The augmentation function is called with additional keyword arguments.

    Parameters:
    input_folder (str): Path to the folder containing the images to augment.
    augmentation_type (str): The type of augmentation to apply. Currently supported: gaussian, random_erase
    output_folder (Optional[str]): Path to the folder where augmented images and label files will be saved.
    **kwargs: Arbitrary keyword arguments passed to the augmentation function.

    Returns:
    None
    """
    if augmentation_type not in augmentation_funcs:
        console.print(
            f"Error Augmentation type `{augmentation_type}` is not supported",
            style=ERROR_STYLE,
        )
        raise ValueError(f"Augmentation type {augmentation_type} is not supported.")

    start_time = time.time()

    augmentation_func = augmentation_funcs[augmentation_type]

    if output_folder is None:
        output_folder = "pythopix_results/augmentation"
        count = 1
        while os.path.exists(output_folder):
            output_folder = f"pythopix_results/augmentation_{count}"
            count += 1

    os.makedirs(output_folder, exist_ok=True)

    for image_path in tqdm.tqdm(
        glob.glob(os.path.join(input_folder, "*.[jp][pn]g")), desc="Augmenting images"
    ):
        augmented_image = augmentation_func(image_path, **kwargs)

        base_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, base_name)
        cv2.imwrite(output_image_path, augmented_image)

        label_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(label_path):
            output_label_path = os.path.join(
                output_folder, os.path.basename(label_path)
            )
            shutil.copy(label_path, output_label_path)
    end_time = time.time()

    console.print(
        f"Successfully augmented images in {round(end_time-start_time,2)} seconds",
        style=SUCCESS_STYLE,
    )


def cut_images(
    input_folder: str,
    output_folder: str = "pythopix_results/cuts",
    num_images: int = 20,
) -> None:
    """
    Cuts out and saves bounding box regions from images based on YOLO format annotations.

    This function processes images in a specified input folder, reads their corresponding YOLO
    annotation files, and cuts out the annotated regions. The cropped images are saved in a given
    output folder, with each image named as 'cutout_{class}_{serial_number}.png'. It handles images
    with multiple bounding boxes and skips images without corresponding label files.

    Args:
    input_folder (str): Path to the folder containing images and their YOLO annotation files.
    output_folder (str, optional): Path to the folder where cropped images will be saved.
                                   Defaults to 'pythopix/cuts'.
    num_images (int, optional): Number of images to process. Defaults to 20.

    Returns:
    None
    """
    start_time = time.time()

    output_folder = get_unique_folder_name(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_processed = 0
    serial_number = 0

    for filename in tqdm.tqdm(os.listdir(input_folder), desc="Cutting images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            label_path = image_path.replace(".jpg", ".txt").replace(".png", ".txt")

            if not os.path.exists(label_path) or images_processed >= num_images:
                continue

            image = cv2.imread(image_path)
            height, width, _ = image.shape

            with open(label_path, "r") as file:
                for line in file:
                    class_id, x_center, y_center, bbox_width, bbox_height = [
                        float(x) for x in line.split()
                    ]
                    class_id = int(class_id)

                    x = int((x_center - bbox_width / 2) * width)
                    y = int((y_center - bbox_height / 2) * height)
                    w = int(bbox_width * width)
                    h = int(bbox_height * height)

                    cropped_image = image[y : y + h, x : x + w]

                    output_filename = f"cutout_{class_id}_{serial_number}.png"
                    cv2.imwrite(
                        os.path.join(output_folder, output_filename), cropped_image
                    )
                    serial_number += 1

            images_processed += 1

    end_time = time.time()

    console.print(
        f"Successfully cutout images in {round(end_time-start_time,2)} seconds",
        style=SUCCESS_STYLE,
    )


def make_backgrounds(
    input_folder: str,
    output_folder: str = "pythopix_results/backgrounds",
    max_backgrounds=None,
) -> None:
    """
    Copies a specified number of images without corresponding YOLO label files from the input folder
    to an output folder. If the number is not specified, all found background images are copied.

    This function processes images in a specified input folder and checks for the existence of
    corresponding YOLO annotation files. Images without a label file are considered as backgrounds
    and are copied to the specified output folder, up to a maximum number if specified.

    Args:
    input_folder (str): Path to the folder containing images and potentially their YOLO annotation files.
    output_folder (str, optional): Path to the folder where background images will be saved.
                                   Defaults to 'pythopix_results/backgrounds'.
    max_backgrounds (int, optional): Maximum number of background images to copy. If None, all found
                                     backgrounds will be copied. Defaults to None.

    Returns:
    None
    """
    start_time = time.time()

    output_folder = get_unique_folder_name(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    backgrounds_copied = 0

    for filename in tqdm.tqdm(os.listdir(input_folder), desc="Making backgrounds"):
        if (max_backgrounds is not None) and (backgrounds_copied >= max_backgrounds):
            break

        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            label_path = image_path.replace(".jpg", ".txt").replace(".png", ".txt")

            if not os.path.exists(label_path):
                shutil.copy(image_path, output_folder)
                backgrounds_copied += 1

    end_time = time.time()

    console.print(
        f"Successfully made backgrounds in {round(end_time-start_time,2)} seconds",
        style=SUCCESS_STYLE,
    )


def extract_class_from_filename(filename: str) -> int:
    """Extracts the class ID from the filename."""
    parts = filename.split("_")
    if len(parts) >= 3 and parts[0] == "cutout":
        return int(parts[1])
    return -1  # Invalid class ID


def make_mosaic_images(
    cutouts_folder: str,
    backgrounds_folder: str,
    output_folder: str = "pythopix_results/mosaic_images",
    num_images: int = 20,
    cutouts_range: Tuple[int, int] = (1, 3),
) -> None:
    """
    Creates mosaic images by superimposing cutout images onto background images.

    This function selects a random background image and a random number of cutout images.
    The cutout images are then placed at random locations in the lower half of the background image.
    For each mosaic image created, a corresponding YOLO format label file is also generated,
    containing the class and bounding box coordinates of each inserted cutout image.

    The class of each cutout image is determined from its filename, which is expected to be in
    the format 'cutout_{class}_{serial_number}.png'.

    Args:
    cutouts_folder (str): Path to the folder containing cutout images.
    backgrounds_folder (str): Path to the folder containing background images.
    output_folder (str): Path to the folder where the mosaic images and their label files will be saved.
                         Defaults to 'pythopix_results/mosaic_images'.
    num_images (int): Number of mosaic images to create. Defaults to 20.
    cutouts_range (Tuple[int, int]): The range (inclusive) of the number of cutouts to be placed on each background.
                                     Defaults to (1, 3).

    Returns:
    None
    """
    start_time = time.time()

    output_folder = get_unique_folder_name(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for i in tqdm.tqdm(range(num_images), desc="Making mosaics"):
        background_image_name = random.choice(os.listdir(backgrounds_folder))
        background_image_path = os.path.join(backgrounds_folder, background_image_name)
        background_image = cv2.imread(background_image_path)

        height, width, _ = background_image.shape
        num_cutouts = random.randint(*cutouts_range)
        placed_cutouts = []
        cutout_files = get_random_files(cutouts_folder, num_cutouts)

        label_content = []

        for cutout_file in cutout_files:
            cutout_path = os.path.join(cutouts_folder, cutout_file)
            cutout_image = cv2.imread(cutout_path)
            cutout_height, cutout_width, _ = cutout_image.shape

            if cutout_height > height // 2 or cutout_width > width:
                continue  # Skip this cutout as it's too large

            attempts = 0
            while attempts < 10:
                x_pos = random.randint(0, width - cutout_width)
                y_pos = random.randint(height // 2, height - cutout_height)

                overlap = any(
                    check_overlap_and_area(
                        x_pos, y_pos, cutout_width, cutout_height, other
                    )
                    for other in placed_cutouts
                )

                if not overlap:
                    break
                attempts += 1

            if attempts < 10:
                cutout_image = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
                cutout_height, cutout_width, num_channels = cutout_image.shape

                if num_channels == 4:  # Image has an alpha channel
                    alpha_s = cutout_image[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        background_image[
                            y_pos : y_pos + cutout_height,
                            x_pos : x_pos + cutout_width,
                            c,
                        ] = (
                            alpha_s * cutout_image[:, :, c]
                            + alpha_l
                            * background_image[
                                y_pos : y_pos + cutout_height,
                                x_pos : x_pos + cutout_width,
                                c,
                            ]
                        )
                else:  # Image does not have an alpha channel
                    background_image[
                        y_pos : y_pos + cutout_height, x_pos : x_pos + cutout_width
                    ] = cutout_image
                placed_cutouts.append((x_pos, y_pos, cutout_width, cutout_height))

            class_id = extract_class_from_filename(cutout_file)
            x_center = (x_pos + cutout_width / 2) / width
            y_center = (y_pos + cutout_height / 2) / height
            bbox_width = cutout_width / width
            bbox_height = cutout_height / height
            label_content.append(
                f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}"
            )

        merged_image_name = f"merged_image_{i}.png"
        cv2.imwrite(os.path.join(output_folder, merged_image_name), background_image)

        with open(
            os.path.join(output_folder, merged_image_name.replace(".png", ".txt")), "w"
        ) as label_file:
            label_file.write("\n".join(label_content))

    end_time = time.time()

    console.print(
        f"Successfully made mosaics in {round(end_time-start_time,2)} seconds",
        style=SUCCESS_STYLE,
    )
