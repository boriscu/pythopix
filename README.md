# pythopix

![pythopix Logo](https://raw.githubusercontent.com/boriscu/pythopix/main/docs/pythopix.png)

PythoPix is a Python library designed for evaluating and analyzing image datasets using YOLO models. It extends beyond simple data analysis to encompass object detection, comparison of detection results, label handling, and model operations, making it a comprehensive tool for both experienced users and beginners in YOLO model applications.

## Features

- **Dataset Evaluation**: Robust evaluation of image datasets using YOLO models, with options for detailed analysis.
- **Model Operations**: Processing images with YOLO models, calculating metrics like IoU, and managing image segregation for augmentation.
- **Label Operations**: Extensive tools for handling, converting, and saving label files in different formats, facilitating versatile dataset management.
- **File Operations**: Efficient handling and saving of predictions and analysis results, crucial for post-analysis data management.
- **Comparison Tools**: Functions for comparing original and predicted labels, aiding in the visual assessment of model accuracy.
- **Data Handling and Export**: Facilitating the export of image analysis results to CSV, enabling easy data sharing and record-keeping.

## Installation

Install pythopix using pip:

```bash
pip install pythopix
```

## Usage

### Evaluating an Image Dataset

```python
from pythopix import evaluate_dataset

evaluate_dataset(
    test_images_folder='/path/to/your/image/dataset',
    model_path='/path/to/your/yolo/model',  # Optional
    num_images=10,  # Optional
    verbose=True,  # Optional
    print_results=True,  # Optional
    copy_images=True  # Optional
)
```

### Comparing Labels

```python
from pythopix.comparison import compare_labels

compare_labels(
    image_path='path/to/image.png',
    predicted_label_path='path/to/predicted/label.txt',
    original_label_path='path/to/original/label.txt',  # Optional
    show=True,  # Optional
    save_fig=False  # Optional
)
```

### Handling Data and Labels

```python
from pythopix.labels_operations import extract_label_files, convert_txt_to_json_labels

# Extract label files
label_files = extract_label_files('/path/to/label_folder')

# Convert TXT to JSON labels
convert_txt_to_json_labels(label_files, label_mapping={0: "Label1", 1: "Label2"})
```

## Requirements

- Python 3.x
- PyTorch
- tqdm
- Ultralytics YOLO
- OpenCV
- matplotlib

## Contributing

Contributions to pythopix are welcome! Please refer to our contribution guidelines for details on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
