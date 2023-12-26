![pythopix Logo](https://raw.githubusercontent.com/boriscu/pythopix/main/docs/pythopix.png)

# pythopix

PythoPix is a Python library designed for evaluating image datasets using YOLO models. It simplifies image data analysis and object detection, making it accessible even for users without a pre-trained YOLO model. The core feature of PythoPix is its ability to re-evaluate the training dataset with the trained model, highlighting false positives and negatives. This functionality is crucial for identifying potential errors or biases in the original dataset, allowing users to understand where additional data augmentation or correction might be necessary.

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

If you don't specify a `model_path`, pythopix will automatically use a default YOLO model.

### Additional Functionalities

pythopix also provides additional utility functions for custom data handling and processing. Refer to our [documentation](https://github.com/boriscu/pythopix/blob/main/docs/DOCUMENTATION.md) for detailed information on these utilities.

## Requirements

- Python 3.x
- PyTorch
- tqdm
- Ultralytics YOLO

## Contributing

Contributions to pythopix are welcome! Please refer to our contribution guidelines for details on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
