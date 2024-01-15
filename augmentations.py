import cv2
import numpy as np
import os
import glob
import shutil
import tqdm

from .theme import console, SUCCESS_STYLE, ERROR_STYLE


def gaussian_noise(
    image_path: str, sigma: float = 25, frequency: float = 1.0
) -> np.ndarray:
    """
    Adds Gaussian noise to an image.

    Parameters:
    image_path (str): The file path to the input image.
    sigma (float): The standard deviation of the Gaussian noise. Higher values mean more intense noise.
    frequency (float): The frequency of applying the noise. A value of 1.0 applies noise to every pixel,
                       while lower values apply it more sparsely.

    Returns:
    np.ndarray: The image with Gaussian noise added.

    Raises:
    FileNotFoundError: If the image at the specified path is not found.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # Generate Gaussian noise
    h, w, c = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (h, w, c)) * frequency
    gauss = gauss.reshape(h, w, c)

    # Add the Gaussian noise to the image
    noisy_image = image + gauss

    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# Available augmentation functions
augmentation_funcs = {"gaussian": gaussian_noise}


def apply_augmentations(
    input_folder: str, augmentation_type: str, output_folder: str = None
):
    """
    Applies a specified type of augmentation to all images in a specified folder and saves the results along with their
    corresponding label files to an output folder.

    Parameters:
    input_folder (str): Path to the folder containing the images to augment.
    augmentation_type (str): The type of augmentation to apply. Currently supports:
                             - "gaussian": Applies Gaussian noise to the images.
    output_folder (str, optional): Path to the folder where augmented images and label files will be saved. If not
                                   specified, defaults to 'pythopix_results/augmentation' or a variation if it already exists.

    Returns:
    None
    """
    if augmentation_type not in augmentation_funcs:
        console.print(
            f"Error Augmentation type `{augmentation_type}` is not supported",
            style=ERROR_STYLE,
        )
        raise ValueError(f"Augmentation type '{augmentation_type}' is not supported.")

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
        augmented_image = augmentation_func(image_path)

        base_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, base_name)
        cv2.imwrite(output_image_path, augmented_image)

        label_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(label_path):
            output_label_path = os.path.join(
                output_folder, os.path.basename(label_path)
            )
            shutil.copy(label_path, output_label_path)

    console.print(
        "Successfully augmented images",
        style=SUCCESS_STYLE,
    )
