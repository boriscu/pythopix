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

- **Description**: Processes an image using the YOLO model for predictions and calculates related metrics.
- **Parameters**:
  - `image_path (str)`: Path of the image to process.
  - `model (YOLO)`: YOLO model for image processing.
  - `verbose (bool)`: Flag for verbose output. Defaults to False.
- **Returns**:
  - `ImageData`: Data about the image, including false positives, false negatives, and box loss.
- **Usage**:

  ```python
  from pythopix.model_operations import process_image

  image_data, predictions = process_image("path/to/image.jpg", model)
  ```
