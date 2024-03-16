from typing import Tuple, List
import random
import cv2
import numpy as np
import os
import glob
import shutil
import torch
import torchvision.utils as vutils
import tqdm
import time

from .labels_operations import convert_to_pixels, read_labels
from .theme import console, SUCCESS_STYLE, ERROR_STYLE
from .utils import get_unique_folder_name, get_random_files, check_overlap_and_area
from PIL import Image, ImageOps


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

        gauss = np.random.normal(mean, sigma, (h, w, c)) * frequency
        gauss = gauss.reshape(h, w, c)

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
    input_folder: str,
    augmentation_type: str,
    output_folder: str = "pythopix_results/augmentations",
    **kwargs,
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


def generate_fake_image(
    generator: torch.nn.Module,
    nz: int = 100,
    save: bool = False,
    output_dir: str = "pythopix_results/dcgan_fake_images",
    seed: int = 999,
) -> torch.Tensor:
    """
    Generates a single fake image using a trained Generator.

    This function also normalizes the generated image's pixel values to [0, 1] using
    torchvision's make_grid with normalize=True, ensuring the image is properly scaled
    for saving in a standard image format.

    Args:
        generator (torch.nn.Module): The trained Generator model.
        nz (int): Size of the latent z vector.
        save (bool): Whether to save the generated image.
        output_dir (str): Directory to save the generated image.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: The generated fake image, normalized to the range [0, 1].
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = generator(noise)

        # Normalize the image to [0, 1] for visualization and saving
        # Even though make_grid is typically for multiple images, it can be used here for a single image
        # to utilize its normalize=True functionality.
        normalized_image = vutils.make_grid(fake_image, normalize=True)

        if save:
            os.makedirs(output_dir, exist_ok=True)
            # Save the normalized image
            save_path = os.path.join(output_dir, "fake_image.png")
            vutils.save_image(normalized_image, save_path)

        return fake_image


def generate_fake_images(
    generator: torch.nn.Module,
    nz: int = 100,
    num_images: int = 128,
    save: bool = False,
    output_dir: str = "pythopix_results/dcgan_fake_images",
    seed: int = 999,
) -> None:
    """
    Generates multiple fake images using a trained Generator and saves them individually.

    Args:
        generator (torch.nn.Module): The trained Generator model.
        nz (int): Size of the latent z vector.
        num_images (int): Number of fake images to generate.
        save (bool): Whether to save the generated images.
        output_dir (str): Directory to save the generated images.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_images, nz, 1, 1, device=device)
        fake_images = generator(noise)

        if save:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(fake_images):
                # Ensure img is 4D (batch_size=1, channels, height, width)
                img_4d = img.unsqueeze(0)  # Adds a batch dimension
                # Normalize and save the image
                save_path = os.path.join(output_dir, f"fake_image_{i}.png")
                vutils.save_image(img_4d, save_path, normalize=True)


def augment_images_with_gan(
    input_folder_path: str,
    generated_images_dir: str,
    allowed_classes: List[int] = [0],
    threshold_width: int = 50,
    threshold_height: int = 50,
    num_of_images: int = 200,
    output_folder: str = "pythopix_results/gan_augmentations",
    small_labels: bool = False,
    extra_area_percentage: int = 10,
) -> None:
    """
    Augments images by replacing selected labels with random images from a specified directory.

    Args:
        input_folder_path (str): Path to the input folder containing images and YOLO labels.
        generated_images_dir (str): Path to the folder containing generated images for augmentation.
        allowed_classes (List[int]): List of class IDs that are allowed to be augmented.
        threshold_width (int): Minimum width in pixels for a label to be augmented.
        threshold_height (int): Minimum height in pixels for a label to be augmented.
        num_of_images (int): Number of images to randomly select and augment.
        output_folder (str): Path to the folder where augmented images will be saved.
        small_labels (bool): If True, crops the generated images by a specified percentage before resizing.
        extra_area_percentage (int): Percentage of the generated image size to be cropped.

    Returns:
        None
    """

    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(".png")]
    selected_images = random.sample(image_files, min(num_of_images, len(image_files)))

    for image_name in tqdm.tqdm(selected_images, desc="Augmenting images"):
        image_path = os.path.join(input_folder_path, image_name)
        label_path = image_path.replace(".png", ".txt")

        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            continue

        with Image.open(image_path).copy() as img:
            img_width, img_height = img.size
            labels = read_labels(label_path)

            for class_id, label in labels:
                if class_id not in allowed_classes:
                    continue

                pixel_label = convert_to_pixels(label, img_width, img_height)
                x1, y1, w, h = pixel_label

                if w > threshold_width and h > threshold_height:
                    generated_images = [
                        f
                        for f in os.listdir(generated_images_dir)
                        if f.endswith(".png")
                    ]

                    random_image_name = random.choice(generated_images)
                    random_image_path = os.path.join(
                        generated_images_dir, random_image_name
                    )

                    with Image.open(random_image_path) as fake_image:
                        if small_labels:
                            crop_width = fake_image.width * (
                                1 - extra_area_percentage / 100
                            )
                            crop_height = fake_image.height * (
                                1 - extra_area_percentage / 100
                            )
                            left = (fake_image.width - crop_width) / 2
                            top = (fake_image.height - crop_height) / 2
                            right = left + crop_width
                            bottom = top + crop_height
                            fake_image = fake_image.crop((left, top, right, bottom))

                        resized_fake_image = fake_image.resize((w, h))
                        img.paste(resized_fake_image, (x1, y1))

        output_path = os.path.join(output_folder, image_name)
        os.makedirs(output_folder, exist_ok=True)
        img.save(output_path)


def augment_image_with_gan(
    input_image_path: str,
    generated_images_dir: str,
    allowed_classes: List[int] = [0],
    threshold_width: int = 50,
    threshold_height: int = 50,
    output_folder: str = "pythopix_results/gan_augmentations",
    small_labels: bool = False,
    extra_area_percentage: int = 10,
) -> None:
    """
    Augments an image by replacing selected labels with random images from a specified directory.

    Args:
        input_image_path (str): Path to the input image and YOLO label.
        generated_images_dir (str): Path to the folder containing generated images for augmentation.
        allowed_classes (List[int]): List of class IDs that are allowed to be augmented.
        threshold_width (int): Minimum width in pixels for a label to be augmented.
        threshold_height (int): Minimum height in pixels for a label to be augmented.
        output_folder (str): Path to the folder where augmented images will be saved.
        small_labels (bool): If True, crops the generated images by a specified percentage before resizing.
        extra_area_percentage (int): Percentage of the generated image size to be cropped.

    Returns:
        None
    """

    image_path = input_image_path
    label_path = image_path.replace(".png", ".txt")

    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return

    with Image.open(image_path).copy() as img:
        img_width, img_height = img.size
        labels = read_labels(label_path)

        for class_id, label in labels:
            if class_id not in allowed_classes:
                continue

            pixel_label = convert_to_pixels(label, img_width, img_height)
            x1, y1, w, h = pixel_label

            if w > threshold_width and h > threshold_height:
                generated_images = [
                    f for f in os.listdir(generated_images_dir) if f.endswith(".png")
                ]

                random_image_name = random.choice(generated_images)
                random_image_path = os.path.join(
                    generated_images_dir, random_image_name
                )

                with Image.open(random_image_path) as fake_image:
                    if small_labels:
                        crop_width = fake_image.width * (
                            1 - extra_area_percentage / 100
                        )
                        crop_height = fake_image.height * (
                            1 - extra_area_percentage / 100
                        )
                        left = (fake_image.width - crop_width) / 2
                        top = (fake_image.height - crop_height) / 2
                        right = left + crop_width
                        bottom = top + crop_height
                        fake_image = fake_image.crop((left, top, right, bottom))

                    resized_fake_image = fake_image.resize((w, h))
                    img.paste(resized_fake_image, (x1, y1))

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    os.makedirs(output_folder, exist_ok=True)
    img.save(output_path)


def generate_padded_images(
    input_folder: str,
    output_folder: str = "pythopix_results/padded_images",
    height: int = 1080,
    width: int = 1920,
    padding_type: str = "gray",
) -> None:
    """
    Generates padded images with a specified background and saves them to an output folder along with their YOLO label files.

    Parameters:
    - input_folder: The path to the folder containing the input images.
    - output_folder: The path to the folder where the padded images and YOLO label files will be saved. Defaults to "pythopix_results/padded_images".
    - height, width: The dimensions of the output images.
    - padding_type: The type of padding to apply. If "average", the padding color is the average color of the image. Otherwise, a gray background is used.

    The function iterates through all images in the input folder, centers them on a background of the specified dimensions, and saves the result to the output folder. A YOLO label file is also generated for each image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm.tqdm(os.listdir(input_folder), desc="Padding images"):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if padding_type == "average":
                average_color = np.mean(img, axis=(0, 1))
            else:
                average_color = (128, 128, 128)  # Gray color

            background = np.full((height, width, 3), average_color, dtype=np.uint8)

            y_center = (height - img.shape[0]) // 2
            x_center = (width - img.shape[1]) // 2

            background[
                y_center : y_center + img.shape[0], x_center : x_center + img.shape[1]
            ] = img

            output_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_img_path, background)

            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(output_folder, label_filename)
            with open(label_path, "w") as label_file:
                x_norm = (x_center + img.shape[1] / 2) / width
                y_norm = (y_center + img.shape[0] / 2) / height
                w_norm = img.shape[1] / width
                h_norm = img.shape[0] / height
                label_file.write(f"0 {x_norm} {y_norm} {w_norm} {h_norm}\n")


def adjust_hsv(
    image_path: str,
    h_perc: float = 0.015,
    s_perc: float = 0.7,
    v_perc: float = 0.4,
    increase: bool = True,
    save: bool = False,
) -> None:
    """
    Adjust the Hue, Saturation, and Value (Brightness) of an image based on a percentile increase or decrease
    from the original values. Save each separately adjusted image and the final image with all adjustments applied.

    Parameters:
    - image_path (str): Path to the input image in PNG format.
    - h_perc (float): Hue adjustment percentage.
    - s_perc (float): Saturation adjustment percentage.
    - v_perc (float): Value (brightness) adjustment percentage.
    - increase (bool): Specifies whether to increase (True) or decrease (False) the HSV values.
    - save (bool): If True, save each adjusted image and the final image in 'pythopix_results/augmentations'.

    Returns:
    - final_image (numpy.ndarray): The final image with all adjustments (H, S, V) applied.
    """
    # Create directory for saving results if it doesn't exist
    save_dir = "pythopix_results/augmentations"
    os.makedirs(save_dir, exist_ok=True)

    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"No image found at {image_path}")

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_factor = 1 + h_perc if increase else 1 - h_perc
    s_factor = 1 + s_perc if increase else 1 - s_perc
    v_factor = 1 + v_perc if increase else 1 - v_perc

    for idx, (factor, name) in enumerate(
        zip([h_factor, s_factor, v_factor], ["hue", "saturation", "value"])
    ):
        adjusted_hsv = hsv_image.copy()
        adjusted_hsv[:, :, idx] *= factor
        adjusted_hsv[:, :, idx] = np.clip(adjusted_hsv[:, :, idx], 0, 255)
        adjusted_bgr = cv2.cvtColor(adjusted_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if save:
            cv2.imwrite(
                os.path.join(
                    save_dir,
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_{name}_adjusted.png",
                ),
                adjusted_bgr,
            )

    hsv_image[:, :, 0] *= h_factor
    hsv_image[:, :, 1] *= s_factor
    hsv_image[:, :, 2] *= v_factor
    hsv_image = np.clip(hsv_image, 0, 255)

    final_image_bgr = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if save:
        cv2.imwrite(
            os.path.join(
                save_dir,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_final_adjusted.png",
            ),
            final_image_bgr,
        )

    return final_image_bgr


def geometric_augmentations(
    image_path: str,
    save: bool = False,
    translate: float = 0.1,
    scale: float = 0.5,
    fliplr: float = 0.5,
) -> Image.Image:
    """
    Applies geometric augmentations (translation, scaling, and horizontal flipping) to an image and saves the augmented images.

    Parameters:
    - image_path (str): Path to the original image.
    - save (bool): Flag to save the intermediate and final augmented images.
    - translate (float): Maximum percentage (0-1) for random translation.
    - scale (float): Maximum percentage (0-1) for random scaling.
    - fliplr (float): Probability (0-1) of applying a horizontal flip.

    Returns:
    - Image.Image: The image after applying all augmentations.
    """

    def random_translate(img: Image.Image, translate_range: float) -> Image.Image:
        max_dx = translate_range * img.width
        max_dy = translate_range * img.height
        dx = int(np.random.uniform(-max_dx, max_dx))
        dy = int(np.random.uniform(-max_dy, max_dy))
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    def random_zoom(
        img: Image.Image, scale_range: float, original_size: Tuple[int, int]
    ) -> Image.Image:
        scale_factor = np.random.uniform(1, 1 + scale_range)
        # Calculate the size of the scaled image
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)

        # Scale the image
        scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Select a random crop area from the scaled image
        left = random.randint(0, new_width - original_size[0])
        top = random.randint(0, new_height - original_size[1])
        right = left + original_size[0]
        bottom = top + original_size[1]
        cropped_img = scaled_img.crop((left, top, right, bottom))

        return cropped_img

    def maybe_fliplr(img: Image.Image, probability: float) -> Image.Image:
        if np.random.random() < probability:
            return ImageOps.mirror(img)
        return img

    output_dir = "pythopix_results/augmentations"
    os.makedirs(output_dir, exist_ok=True)

    original_img = Image.open(image_path)
    augmented_img = original_img.copy()

    if save:
        translated_img = random_translate(original_img, translate)
        translated_img.save(os.path.join(output_dir, "translated.jpg"))

        zoomed_img = random_zoom(
            original_img, scale, (original_img.width, original_img.height)
        )
        zoomed_img.save(os.path.join(output_dir, "zoomed.jpg"))

        flipped_img = maybe_fliplr(original_img, fliplr)
        flipped_img.save(os.path.join(output_dir, "flipped.jpg"))

        augmented_img = random_zoom(
            augmented_img, scale, (original_img.width, original_img.height)
        )
        augmented_img = maybe_fliplr(augmented_img, fliplr)

        augmented_img = random_translate(augmented_img, translate)
        augmented_img.save(os.path.join(output_dir, "all_augmented.jpg"))
    else:
        augmented_img = random_translate(augmented_img, translate)
        augmented_img = random_zoom(
            augmented_img, scale, (original_img.width, original_img.height)
        )
        augmented_img = maybe_fliplr(augmented_img, fliplr)

    return augmented_img


def mixup_augmentations(
    image_path1: str,
    image_path2: str,
    output_dir: str = "pythopix_results/augmentations",
) -> Image.Image:
    """
    Takes two image paths, sets their transparency to 50%, overlays them together,
    and saves the resulting image to a specified directory. Additionally, the overlaid
    image is returned.

    Parameters:
    - image_path1 (str): Path to the first image.
    - image_path2 (str): Path to the second image.
    - output_dir (str): Directory path where the overlaid image will be saved. Defaults to 'pythopix_results/augmentations'.

    Returns:
    - Image.Image: The overlaid image with 50% transparency for both input images.
    """
    image1 = Image.open(image_path1).convert("RGBA")
    image2 = Image.open(image_path2).convert("RGBA")

    image2 = image2.resize(image1.size)

    alpha = 0.5
    image1.putalpha(int(255 * alpha))
    image2.putalpha(int(255 * alpha))

    overlaid_image = Image.alpha_composite(image1, image2)

    import os

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "overlaid_image.png")
    overlaid_image.save(output_path, format="PNG")

    return overlaid_image
