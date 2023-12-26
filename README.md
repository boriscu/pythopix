![pythopix Logo](https://github.com/boriscu/pythopix/blob/main/docs/pythopix.png)

# pythopix

PythoPix is a Python library for evaluating image datasets using YOLO models. It's designed to make the process of image data analysis and object detection straightforward and accessible, even for those who may not have a pre-trained YOLO model.

## Features

- Easy evaluation of image datasets for object detection.
- Automatic fallback to a default YOLO model if a custom model is not provided.
- Options for detailed analysis, including metrics export and image segregation.

## Installation

You can install pythopix directly using pip:

```bash
pip install pythopix
```

## Usage

### Evaluating an Image Dataset

To evaluate an image dataset, you need to provide the path to your dataset. Optionally, you can also specify a custom model path, number of images for additional augmentation, and other parameters.

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

If you don't specify a `model_path`, OptiPyth will automatically use a default YOLO model.

### Additional Functionalities

OptiPyth also provides additional utility functions for custom data handling and processing. Refer to our [documentation](#) for detailed information on these utilities.

## Requirements

- Python 3.x
- PyTorch
- tqdm
- Ultralytics YOLO

These dependencies are automatically installed when you install pythopix via pip.

## Contributing

Contributions to pythopix are welcome! Please refer to our contribution guidelines for details on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
