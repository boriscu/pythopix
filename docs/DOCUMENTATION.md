# Modules and Functions

## Dataset Evaluation

### `evaluate_dataset`

- **Description**: Evaluates an image dataset using a YOLO model. It can use either a provided model or a default YOLO model.
- **Parameters**:
  - `test_images_folder (str)`: Path to the test images.
  - `model_path (Optional[str])`: Path to the model weights file. Defaults to `None`, using the default YOLO model.
  - `num_images (int)`: Number of images for additional augmentation. Defaults to `100`.
  - `verbose (bool)`: Enable verbose output. Defaults to `False`.
  - `print_results (bool)`: Print sorted image data results. Defaults to `False`.
  - `copy_images (bool)`: Copy images for additional augmentation. Defaults to `False`.
- **Returns**:
  - `List[dict]`: A list of dictionaries containing sorted image data based on the evaluation.
- **Usage**:

  ```python
  from pythopix.dataset_evaluation import evaluate_dataset
  evaluate_dataset("path/to/test_images", model_path="path/to/model", num_images=50, verbose=True)
  ```

### `plot_bb_distribution`

- **Description**: Plots the distribution of bounding box areas from a list of YOLO label file paths and optionally saves the plot.
- **Parameters**:
  - `label_paths (List[str])`: A list of paths to YOLO label files.
  - `save (bool)`: If True, saves the plot to 'pythopix_results/bbox_distribution.png'. Defaults to False.
- **Returns**:
  - None
- **Usage**:

  ```python
  from pythopix.dataset_evaluation import plot_bb_distribution

  label_files = extract_label_files('/path/to/label_folder')
  plot_label_distribution(label_files, save=True)
  ```

### `calculate_segmented_metrics`

- **Description**: Processes a folder of images and calculates average detection metrics for segmented bounding boxes based on their sizes. The function divides bounding boxes into specified segments and computes average false positives, false negatives, and box loss for each segment.
- **Parameters**:
  - `folder_path (str)`: Path to the folder containing images and corresponding YOLO label files.
  - `model (YOLO, optional)`: An instance of the YOLO model. If None, the model is loaded from the specified `model_path`.
  - `model_path (str, optional)`: Path to load the YOLO model from, used if `model` is None.
  - `segment_number (int, optional)`: Number of segments to divide the bounding boxes into based on their sizes. Defaults to 4.
- **Returns**:
  - `Dict[str, Tuple[float, float, float]]`: A dictionary where each key is a segment range (e.g., '0.00-0.25'), and the value is a tuple containing the average false positives, false negatives, and box loss for that segment.
- **Usage**:

  ```python
  from pythopix.metrics import calculate_segmented_metrics

  metrics = calculate_segmented_metrics('/path/to/folder', model=my_model, segment_number=4)
  ```

### `plot_metrics_by_segment`

- **Description**: Plots and optionally saves three bar charts for given segmented metrics. There is one chart each for false positives, false negatives, and box loss, illustrating the performance across different bounding box size segments.
- **Parameters**:
  - `metrics_by_segment (Dict[str, Tuple[float, float, float]])`: A dictionary with segment ranges as keys and tuples of metrics as values.
  - `save (bool, optional)`: If True, saves the plots to the 'pythopix_results' folder. Defaults to False.
- **Usage**:

  ```python
  from pythopix.visualization import plot_metrics_by_segment

  plot_metrics_by_segment(metrics, save=True)
  ```

### `save_segmented_metrics_to_csv`

- **Description**: Saves the segmented metrics data to a CSV file for easy analysis and record-keeping. The CSV file will have segments as columns and metrics (false positives, false negatives, box loss) as rows.
- **Parameters**:
  - `metrics_by_segment (Dict[str, Tuple[float, float, float]])`: Metrics data segmented by bounding box sizes.
- **Usage**:

  ```python
  from pythopix.dataset_evaluation import save_segmented_metrics_to_csv

  save_segmented_metrics_to_csv(metrics)
  ```

### `visualize_bounding_boxes`

- **Description**:

  - This function serves to display or save an image with its bounding boxes as defined in its corresponding YOLO label file. It's designed for visually verifying the accuracy of bounding box annotations. The function reads the label file, which should have the same name as the image file but with a `.txt` extension. It then draws the bounding boxes on the image. Depending on the `save_fig` parameter, it either displays the image with bounding boxes or saves it to a specified folder.

- **Parameters**:

  - `image_path (str)`: The file path to the input image. The function expects to find a YOLO format label file with the same base name and in the same directory as this image file.
  - `save_fig (bool, optional)`: If set to `True`, the function saves the image with bounding boxes to the `pythopix_results/figs` folder instead of displaying it. Defaults to `False`.

- **Returns**:

  - None: Based on the `save_fig` parameter, the function either displays a window with the image and overlaid bounding boxes or saves the image to a folder.

- **Usage**:

  - Examples of using the `visualize_bounding_boxes` function:

    ```python
    from pythopix.dataset_evaluation import visualize_bounding_boxes

    # To display the image with bounding boxes
    visualize_bounding_boxes("path/to/image.jpg")

    # To save the image with bounding boxes
    visualize_bounding_boxes("path/to/image.jpg", save_fig=True)
    ```

### `plot_label_size_distribution`

- **Description**: Plots and optionally saves the distribution of label widths and heights in pixels, as extracted from YOLO label files. The function reads each label file, finds the corresponding image file to obtain actual dimensions, and then calculates the pixel dimensions of the bounding boxes. It generates two subplots: one for the width distribution and another for the height distribution.
- **Parameters**:
  - `input_folder (str)`: Path to the folder containing images and their corresponding YOLO label files.
  - `save (bool, optional)`: If set to `True`, the plot is saved to 'pythopix_results/figs/label_size_distribution.png'. Defaults to `False`.
- **Returns**:
  - None: Based on the `save` parameter, the function either displays the plot or saves it to a specified directory.
- **Usage**:

  ```python
  from pythopix.dataset_evaluation import plot_label_size_distribution

  # To display the plot
  plot_label_size_distribution('path/to/input_folder')

  # To save the plot
  plot_label_size_distribution('path/to/input_folder', save=True)
  ```

### `plot_label_ratios`

- **Description**: This function plots the distribution of label aspect ratios (width divided by height) for a set of images using their corresponding YOLO label files. It calculates the aspect ratio for each bounding box in the label files, then generates and displays a histogram of these ratios. The function can optionally save the histogram plot to a file.

- **Parameters**:

  - `input_folder (str)`: Path to the folder containing images and their corresponding YOLO label files.
  - `save (bool, optional)`: If set to `True`, the function saves the plot as 'label_ratios.png' in the 'pythopix_results/figs' directory. Defaults to `False`.
  - `show (bool)`: If `True`, displays the plot. Defaults to `True`.

- **Returns**:

  - None: The function displays or saves the plot based on the `save` and `show` parameters but does not return any value.

- **Usage**:

  ```python
  from pythopix.dataset_evaluation import plot_label_ratios

  # To display the plot
  plot_label_ratios('path/to/input_folder')

  # To save the plot
  plot_label_ratios('path/to/input_folder', save=True)
  ```

## Data Handling

This module focuses on handling and exporting image data, particularly useful in post-analysis data management.

### `export_to_csv`

- **Description**: Exports a list of `ImageData` objects to a CSV file.
- **Parameters**:
  - `image_data_list (List[ImageData])`: List of `ImageData` objects to export.
  - `filename (str)`: Name of the CSV file to create. Defaults to "image_data_results.csv".
- **Usage**:

  ```python
  from pythopix.data_handling import export_to_csv

  # Assuming image_data_list is a list of ImageData objects
  export_to_csv(image_data_list, "output_filename.csv")
  ```

## File Operations

The `file_operations` module in PythoPix library provides functionality for handling and saving predictions from image analysis.

### `save_predictions`

- **Description**: Saves the predicted bounding boxes and classes for an image to a text file.
- **Parameters**:
  - `image_path (str)`: Path of the image being processed.
  - `predicted_boxes (List[Tensor])`: List of predicted bounding boxes, each in `[x1, y1, x2, y2]` format.
  - `predicted_classes (List[int])`: List of classes corresponding to each bounding box.
- **Behavior**: The function generates a text file in the `pythopix_results/additional_augmentation` folder, containing the formatted predictions for each bounding box associated with the image.
- **Usage**:

  ```python
  from pythopix.file_operations import save_predictions

  save_predictions("path/to/image.jpg", predicted_boxes, predicted_classes)
  ```

## Labels Operations

The `labels_operations` module in PythoPix provides tools for extracting, converting, and saving label files in various formats.

### `extract_label_files`

- **Description**: Extracts the full paths of label files (`.txt` or `.json`) from a specified folder.
- **Parameters**:
  - `source_folder (str)`: The folder to extract label files from.
  - `label_type (str)`: The type of label files to extract ('txt' or 'json'). Defaults to 'txt'.
- **Usage**:

  ```python
  from pythopix.labels_operations import extract_label_files

  txt_label_files = extract_label_files("path/to/folder", "txt")
  json_label_files = extract_label_files("path/to/folder", "json")
  ```

### `save_extracted_labels`

- **Description**: Saves label files to a specified destination folder.
- **Parameters**:
  - `label_files (List[str])`: Paths of label files to be saved.
  - `destination_folder (str)`: Destination folder. Defaults to 'pythopix_results/txt_labels'.
- **Usage**:

  ```python
  from pythopix.labels_operations import save_extracted_labels

  save_extracted_labels(label_files, "path/to/destination")
  ```

### `convert_txt_to_json_labels`

- **Description**: Converts `.txt` label files to `.json` format.
- **Parameters**:
  - `txt_label_files (List[str])`: Paths of `.txt` label files.
  - `label_mapping (Dict[int, str])`: Mapping of numeric labels to string labels.
  - `destination_folder (str)`: Destination folder for `.json` files. Defaults to 'pythopix_results/json_labels'.
- **Usage**:

  ```python
  from pythopix.labels_operations import convert_txt_to_json_labels

  label_mapping = {0: 'Label1', 1: 'Label2'}
  convert_txt_to_json_labels(txt_label_files, label_mapping)
  ```

### `convert_json_to_txt_labels`

- **Description**: Converts `.json` label files to `.txt` format.
- **Parameters**:
  - `json_label_files (List[str])`: Paths of `.json` label files.
  - `base_destination_folder (str)`: Base path for the destination folder. Defaults to 'pythopix_results/txt_labels'.
- **Usage**:

  ```python
  from pythopix.labels_operations import convert_json_to_txt_labels

  convert_json_to_txt_labels(json_label_files)
  ```

### `extract_label_sizes`

- **Description**: Extracts the widths and heights of bounding boxes in pixels from a list of YOLO label files.

- **Parameters**:

  - `label_files (List[str])`: A list of file paths to YOLO label files.

- **Returns**:

  - `Tuple[List[int], List[int]]`: Two lists containing the widths and heights of the bounding boxes in pixels, respectively.

- **Usage**:

  ```python
  from pythopix.labels_operations import extract_label_sizes

  widths, heights = extract_label_sizes(label_files)
  ```

### `filter_and_resize_labels`

- **Description**: This function processes images in a given input folder. It filters labels based on non-overlapping criteria, minimum size, and specified allowed classes. Then, it resizes and saves the labels to the output folder.

- **Parameters**:

  - `input_folder (str)`: Path to the folder containing images and their corresponding YOLO label files.
  - `output_folder (str)`: Path to the folder where the processed labels will be saved. Defaults to "pythopix_results/filtered_labels".
  - `min_width (int)`: Minimum width in pixels for the labels to be considered.
  - `min_height (int)`: Minimum height in pixels for the labels to be considered.
  - `resize_aspect_ratio (Tuple[int, int])`: The aspect ratio to resize the labels to.
  - `alpha (float)`: Threshold for maximum allowed overlap as a fraction of the first rectangle's area.
  - `allowed_classes (Optional[List[int]])`: List of allowed class IDs for filtering labels. If `None`, all classes are processed.

- **Returns**:

  - None

- **Usage**:

  ```python
  from pythopix.labels_operations import filter_and_resize_labels

  # Basic usage with default parameters
  filter_and_resize_labels('path/to/input_folder')

  # Advanced usage with specified parameters
  filter_and_resize_labels('path/to/input_folder', 'path/to/output_folder', 50, 50, (100, 100), 0.2, [0, 1, 2])
  ```

---

### `read_labels`

- **Description**: Reads YOLO label files and returns a list of labels with class IDs and normalized bounding box coordinates.

- **Parameters**:

  - `label_file (str)`: Path to the YOLO label file.

- **Returns**:

  - `List[Tuple[int, Tuple[float, float, float, float]]]`: A list of tuples where each tuple contains a class ID and normalized bounding box coordinates (x_center, y_center, width, height).

- **Usage**:

  ```python
  from pythopix.labels_operations import read_labels

  labels = read_labels('path/to/label_file.txt')
  ```

---

### `convert_to_pixels`

- **Description**: Converts normalized YOLO label coordinates to pixel coordinates based on the image dimensions.

- **Parameters**:

  - `label (Tuple[float, float, float, float])`: Normalized bounding box coordinates (x_center, y_center, width, height).
  - `img_width (int)`: The width of the image in pixels.
  - `img_height (int)`: The height of the image in pixels.

- **Returns**:

  - `Tuple[int, int, int, int]`: The bounding box coordinates in pixel values (x1, y1, width, height).

- **Usage**:

  ```python
  from pythopix.labels_operations import convert_to_pixels

  pixel_coordinates = convert_to_pixels((0.5, 0.5, 0.2, 0.2), 1920, 1080)
  ```

## Utils

### `read_yolo_labels`

- **Description**: Reads YOLO label files and returns a list of labels.
- **Parameters**:
  - `file_path (str)`: Path to the YOLO label file.
- **Returns**:
  - `List[Label]`: A list of Label objects parsed from the file.
- **Usage**:

  ```python
  from pythopix.utils import read_yolo_labels

  labels = read_yolo_labels('/path/to/label_file.txt')
  ```

### `calculate_bb_area`

- **Description**: Calculates the surface area of a bounding box from a Label object.
- **Parameters**:
  - `label (Label)`: A Label object representing the bounding box and class ID.
- **Returns**:

  - `float`: The fractional surface area of the bounding box, as a proportion of the total image area.

- **Usage**:

  ```python
  from pythopix.utils import calculate_bb_area

  area = calculate_bb_area(label)
  ```

### `check_overlap_and_area`

- **Description**:

  - Checks if two rectangular regions overlap and if the overlap exceeds a specified threshold percentage of the area of the first rectangle.

- **Parameters**:

  - `x1 (int)`: X-coordinate of the top-left corner of the first rectangle.
  - `y1 (int)`: Y-coordinate of the top-left corner of the first rectangle.
  - `w1 (int)`: Width of the first rectangle.
  - `h1 (int)`: Height of the first rectangle.
  - `other (Tuple[int, int, int, int])`: A tuple containing the x-coordinate, y-coordinate, width, and height of the second rectangle.
  - `threshold (float, optional)`: Threshold for determining if the overlap is significant, default is set to 20% (0.2).

- **Returns**:

  - `bool`: Returns `True` if there is an overlap that is more than the specified threshold of the area of the first rectangle. Otherwise, it returns `False`.

- **Usage**:

  - Example of using `check_overlap_and_area`:

    ```python
    from pythopix.utils import check_overlap_and_area

    overlap = check_overlap_and_area(10, 20, 100, 100, (50, 60, 100, 100))
    if overlap:
        print("Overlap is more than the threshold.")
    else:
        print("Overlap is within acceptable limits.")
    ```

## Model Operations

The `model_operations` module in PythoPix includes functions for processing images with a YOLO model, calculating Intersection over Union (IoU) for bounding boxes, and segregating images based on the need for additional augmentation.

### `bbox_iou`

- **Description**: Calculates the Intersection over Union (IoU) between two sets of bounding boxes.
- **Parameters**:
  - `box1 (Tensor)`: Bounding boxes as tensors in [x1, y1, x2, y2] format.
  - `box2 (Tensor)`: Bounding boxes to compare against, in the same format.
- **Usage**:

  ```python
  from pythopix.model_operations import bbox_iou

  iou = bbox_iou(tensor_box1, tensor_box2)
  ```

### `segregate_images`

- **Description**: Segregates images into folders based on the need for additional augmentation and saves predicted labels.
- **Parameters**:
  - `image_data_list (List[ImageData])`: Sorted list of `ImageData` objects.
  - `predictions_dict (dict)`: Dictionary with predicted boxes and classes.
  - `num_images (int)`: Number of images to segregate. Defaults to 10.
- **Usage**:

  ```python
  from pythopix.model_operations import segregate_images

  segregate_images(image_data_list, predictions_dict, num_images=10)
  ```

### `process_image`

- **Description**: Processes an image using a YOLO model and calculates accuracy metrics such as false positives, false negatives, and box loss. The function can use a provided model instance or load a model from a specified path.

- **Parameters**:

  - `image_path (str)`: The file path of the image to be processed.
  - `model (YOLO, optional)`: An instance of the YOLO model used for object detection. If `None`, the model is loaded from `model_path`.
  - `model_path (str, optional)`: The file path to load the YOLO model from. This is used if `model` is `None`.
  - `verbose (bool, optional)`: If set to `True`, provides detailed output during the prediction process. Defaults to `False`.

- **Returns**:

  - `Tuple[ImageData, dict]`: A tuple containing:
    - `ImageData`: An object with detailed information about the image, including metrics like false positives, false negatives, and box loss.
    - `dict`: A dictionary containing the predicted bounding boxes and the classes of the detected objects.

- **Usage**:

  ```python
  from your_module import process_image, YOLO

  # If using a pre-loaded model
  model = YOLO("path/to/model")
  image_data, predictions = process_image("path/to/image.jpg", model=model)

  # Or, if loading the model from a path
  image_data, predictions = process_image("path/to/image.jpg", model_path="path/to/model")
  ```

This updated documentation includes the optional parameters and provides examples for different usage scenarios.

## Comparison

The `comparison` module in PythoPix includes functions for comparing original and predicted labels of images, primarily used for visualizing and assessing the performance of object detection models like YOLO.

### `compare_labels`

- **Description**: Compares the original and predicted labels for a single image and displays or saves the comparison.
- **Parameters**:
  - `image_path (str)`: Path to the original image file.
  - `predicted_label_path (str)`: Path to the predicted label file in YOLO format.
  - `original_label_path (Optional[str])`: Path to the original label file. If not provided, it attempts to find a label file with the same name as the image in its directory. Defaults to None.
  - `show (bool)`: Flag to display the plot. Defaults to True.
  - `save (bool)`: Flag to save the figure. Defaults to False.
- **Usage**:

  ```python
  from pythopix.comparison import compare_single_image_labels

  compare_labels('/path/to/image.png', '/path/to/predicted_label.txt', show=True, save=False)
  ```

### `compare_folder_labels`

- **Description**: Compares original images labels in a folder with their corresponding predicted labels and optionally shows or saves the comparisons.
- **Parameters**:
  - `image_folder (str)`: Path to the folder containing images with their original labels.
  - `labels (List[str])`: List of paths to predicted label files.
  - `limit (Optional[int])`: Maximum number of images to process. If None, all images in the folder are processed. Defaults to None.
  - `show (bool)`: Flag to display the comparison plot for each image. Defaults to False.
  - `save (bool)`: Flag to save the comparison plot for each image. Defaults to True.
- **Usage**:

  ```python
  from pythopix.labels_operations import extract_label_files
  from pythopix.comparison import compare_folder_labels

  labels_txt = extract_label_files(
      "pythopix_results/additional_augmentation", label_type="txt"
  )
  compare_folder_labels('/path/to/image_folder', labels=labels_txt, limit=5, show=True, save=True)
  ```

### `yolo_to_bbox`

- **Description**: Converts bounding box data from YOLO format to pixel coordinates.
- **Parameters**:
  - `yolo_data (List[float])`: A list containing four float values representing the bounding box in YOLO format: [x_center, y_center, width, height].
  - `img_width (int)`: The width of the image in pixels.
  - `img_height (int)`: The height of the image in pixels.
- **Returns**:
  - `Tuple[int, int, int, int]`: A tuple of four integer values representing the bounding box in pixel coordinates: (xmin, ymin, xmax, ymax).
- **Usage**:

  ```python
  from pythopix.comparison import yolo_to_bbox

  bbox = yolo_to_bbox([0.5, 0.5, 0.1, 0.1], img_width, img_height)
  ```

### `add_bboxes_to_image`

- **Description**: Draws bounding boxes on an image based on YOLO format label data.
- **Parameters**:
  - `image (List[List[List[int]]])`: The image data in OpenCV format (BGR).
  - `label_file (str)`: Path to the label file containing bounding boxes in YOLO format.
  - `img_width (int)`: The width of the image in pixels.
  - `img_height (int)`: The height of the image in pixels.
- **Returns**:
  - `List[List[List[int]]]`: The image data with bounding boxes drawn, in OpenCV format (BGR).
- **Usage**:

  ```python
  from pythopix.comparison import add_bboxes_to_image

  image_with_boxes = add_bboxes_to_image(image, '/path/to/label_file.txt', img_width, img_height)
  ```

## Image Augmentation

### `apply_augmentations`

- **Description**: Applies a specified type of augmentation to all images in a given folder and saves the results along with their corresponding label files to an output folder. The function supports various augmentation types and allows for flexible parameter specification for each augmentation method.
- **Parameters**:
  - `input_folder (str)`: Path to the folder containing the images to augment.
  - `augmentation_type (str)`: The type of augmentation to apply. Currently supports "gaussian" for Gaussian noise and "random_erase" for random erasing.
  - `output_folder (Optional[str])`: Path to the folder where augmented images and label files will be saved. If not specified, defaults to `pythopix_results/augmentation` or a variation if it already exists.
  - Additional keyword arguments (`**kwargs`) for the specific augmentation function.
- **Returns**:
  - `None`: The function saves the augmented images and label files to the specified folder and does not return any value.
- **Usage**:

  ```python
  from pythopix.image_augmentation import apply_augmentations

  # For Gaussian noise
  apply_augmentations("path/to/input_folder", "gaussian", "path/to/output_folder", sigma=25, frequency=1.0)

  # For Random Erasing
  apply_augmentations(
      "path/to/input_folder",
      "random_erase",
      "path/to/output_folder",
      erasing_prob=0.5,
      area_ratio_range=(0.02, 0.4),
      aspect_ratio_range=(0.3, 3)
  )
  ```

### `gaussian_noise`

- **Description**:

  - Adds Gaussian noise to an image with variable intensity and on a probabilistic basis. This function is particularly useful for augmenting images to improve the robustness and generalization of machine learning models, especially for tasks involving image classification and localization.

- **Parameters**:

  - `image_path (str)`: The file path to the input image.
  - `sigma_range (tuple, optional)`: A tuple specifying the range of standard deviation for the Gaussian noise. The function randomly selects a `sigma` value within this range for each image, allowing for variability in noise intensity. Higher values in the range result in more intense noise. Defaults to `(10, 25)`.
  - `frequency (float, optional)`: The frequency of applying the noise. A value of `1.0` applies noise to every pixel, while lower values apply it more sparsely. Defaults to `1.0`.
  - `noise_probability (float, optional)`: The probability of applying noise to an image. Ranges from `0` (no noise) to `1` (always add noise). Defaults to `0.5`. This allows the model to train on both noisy and clean images.

- **Returns**:

  - `np.ndarray`: The processed image, which may or may not have Gaussian noise added, based on the `noise_probability`.

- **Usage**:

  - Here's an example of how to use the `gaussian_noise` function:

    ```python
    from image_augmentation import gaussian_noise
    # Applying Gaussian noise with a variable sigma and a 50% chance of noise application
    noisy_image = gaussian_noise("path/to/image.jpg", sigma_range=(15, 30), frequency=0.7, noise_probability=0.5)
    ```

### `random_erasing`

- **Description**: Applies Random Erasing augmentation to an image. This function is effective for simulating occlusions and enhancing the robustness of machine learning models, especially in environments where objects might be partially obscured.
- **Parameters**:
  - `image_path (str)`: Path to the input image.
  - `erasing_prob (float)`: Probability of erasing a random patch in the image. Defaults to `0.5`.
  - `area_ratio_range (Tuple[float, float])`: Range of the ratio of the erased area to the total image area. Specifies how large the erased patch can be relative to the image. Defaults to `(0.02, 0.2)`.
  - `aspect_ratio_range (Tuple[float, float])`: Range of the aspect ratio of the erased area. Controls the shape of the erased patch, from narrow to wide. Defaults to `(0.3, 3)`.
- **Returns**:
  - `np.ndarray`: The image with a random patch erased.
- **Usage**:

  ```python
  from pythopix.image_augmentation import random_erasing
  erased_image = random_erasing("path/to/image.jpg", erasing_prob=0.5, area_ratio_range=(0.02, 0.4), aspect_ratio_range=(0.3, 3))
  ```

### `cut_images`

- **Description**:

  - Processes images from a given folder, extracting bounding box regions as defined in corresponding YOLO format annotation files. Each cropped image is saved with a filename indicating its class and a serial number.

- **Parameters**:

  - `input_folder (str)`: Path to the folder with images and YOLO annotation files.
  - `output_folder (str, optional)`: Path for saving the cropped images. Defaults to `"pythopix_results/cuts"`.
  - `num_images (int, optional)`: Number of images to process. Defaults to `20`.

- **Returns**:

  - None: Cropped images are saved in the specified output folder.

- **Usage**:

  - Example of using `cut_images`:

    ```python
    cut_images('path/to/input_images', 'path/to/output_cuts', 15)
    ```

### `make_backgrounds`

- **Description**:

  - Copies images without corresponding YOLO label files from an input folder to an output folder. These images are considered as 'backgrounds'. The function can limit the number of backgrounds copied.

- **Parameters**:

  - `input_folder (str)`: Path to the folder with images and potentially their YOLO annotation files.
  - `output_folder (str, optional)`: Path for saving the background images. Defaults to `"pythopix_results/backgrounds"`.
  - `max_backgrounds (int, optional)`: Maximum number of background images to copy. If unspecified, all backgrounds found will be copied.

- **Returns**:

  - None: Background images are saved in the specified output folder.

- **Usage**:

  - Example of using `make_backgrounds`:

    ```python
    make_backgrounds('path/to/input_images', 'path/to/output_backgrounds', 30)
    ```

### `make_mosaic_images`

- **Description**:

  - Creates mosaic images by overlaying cutout images onto background images. It selects a random background image and places a specified number of cutout images randomly within the lower half of the background. The function ensures that these cutout images do not overlap beyond a specified threshold. It also generates corresponding YOLO format label files for each mosaic, detailing the class and bounding box coordinates of each inserted cutout.

- **Parameters**:

  - `cutouts_folder (str)`: Path to the folder containing cutout images.
  - `backgrounds_folder (str)`: Path to the folder containing background images.
  - `output_folder (str, optional)`: Path for saving the mosaic images and their label files. Defaults to `"pythopix_results/mosaic_images"`.
  - `num_images (int, optional)`: Number of mosaic images to create. Defaults to `20`.
  - `cutouts_range (Tuple[int, int], optional)`: Range of the number of cutouts to be placed on each background, inclusive. Defaults to `(1, 3)`.

- **Returns**:

  - None: The function saves the mosaic images and label files to the specified output folder.

- **Usage**:
  - Example of using `make_mosaic_images`:
    ```python
    make_mosaic_images('path/to/cutouts', 'path/to/backgrounds', 'path/to/output', 10, (1, 5))
    ```

## Models

Models module contains various ml models

### `DCGAN`

- **Description**: Initializes and trains a Deep Convolutional Generative Adversarial Network (DCGAN). The DCGAN consists of a generator and a discriminator that are trained adversarially to generate realistic images.
- **Parameters**:
  - `dataroot (str)`: Root directory for the dataset.
  - `ngpu (int)`: Number of GPUs available. Defaults to 1.
  - `nz (int)`: Size of the latent Z vector. Defaults to 100.
  - `ngf (int)`: Size of feature maps in the generator. Defaults to 128.
  - `ndf (int)`: Size of feature maps in the discriminator. Defaults to 128.
  - `nc (int)`: Number of channels in the images. Defaults to 3.
  - `lr (float)`: Learning rate for optimizers. Defaults to 0.0002.
  - `beta1 (float)`: Beta1 hyperparameter for Adam optimizers. Defaults to 0.5.
  - `batch_size (int)`: Batch size during training. Defaults to 128.
  - `workers (int)`: Number of worker threads for loading the data. Defaults to 2.
  - `num_epochs (int)`: Number of training epochs. Defaults to 5.
- **Methods**:
  - `train`: Trains the DCGAN model. Saves checkpoints and model weights at specified intervals, and returns training statistics including losses and image samples.
- **Usage**:

  ```python
  from pythopix.models import DCGAN

  dcgan = DCGAN(
      ngpu=1,
      nz=100,
      ngf=128,
      ndf=128,
      nc=3,
      lr=0.0002,
      beta1=0.5,
      dataroot="pythopix_results\cuts",
      batch_size=128,
      workers=2,
      num_epochs=5
  )

  dcgan.train()
  ```
