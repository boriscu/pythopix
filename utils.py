from .data_handling import ImageData


def custom_sort_key(image_data: ImageData) -> tuple:
    """
    Custom sorting key function for sorting ImageData objects.

    This function is used to sort ImageData instances based on false positives and box loss,
    with special handling for string values of box_loss.

    Args:
    image_data (ImageData): An instance of ImageData to be sorted.

    Returns:
    tuple: A tuple containing sorting keys.
    """

    if isinstance(image_data.box_loss, str):
        if image_data.box_loss == "Inf":
            return (
                image_data.false_positives,
                image_data.false_negatives,
                float("inf"),
            )
        elif image_data.box_loss == "No GT, FP detected":
            return (
                image_data.false_positives,
                image_data.false_negatives,
                0,
            )  # Assign a specific value for this case
    elif image_data.box_loss is None:
        return (image_data.false_positives, image_data.false_negatives, -float("inf"))
    else:
        return (
            image_data.false_positives,
            image_data.false_negatives,
            image_data.box_loss,
        )
