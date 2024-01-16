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
