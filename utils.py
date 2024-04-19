import cv2
import numpy as np
import torch


def to_tensor(image):
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def to_tensor_lab(
    image,
    mean=torch.tensor([0.5875, 0.5572, 0.5204]),
    std=torch.tensor([0.2818, 0.2785, 0.2905]),
):
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean_expanded = mean.view(1, 3, 1, 1)
    std_expanded = std.view(1, 3, 1, 1)
    tensor = (tensor - mean_expanded) / std_expanded
    return tensor


def threshold(image, threshold_value=250):
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_artifacts(image, min_size=1000):
    image = np.uint8(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )

    output = np.zeros_like(image)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            output[labels == i] = 255

    # If the pixel is 255 in the output image then keep the pixel in the final image
    return np.where(output == 255, image, 0)


def dialate(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
