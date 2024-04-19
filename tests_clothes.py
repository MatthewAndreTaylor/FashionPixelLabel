import torch
import matplotlib.pyplot as plt
from ImageDataset import ImageSet, DeviceDataLoader
from model import DeepLabv3
from trainers import fit
from loss import WeightedCrossEntropyLoss
import numpy as np

image_dir = "data/images"
label_dir = "data/labels/clothes"

MyPeopleSet = ImageSet(image_dir, label_dir, label_type="multi")

# Set the seed for the random split
torch.manual_seed(42)
train, val, test = torch.utils.data.random_split(MyPeopleSet, (0.5, 0.1, 0.4))
train_loader = DeviceDataLoader(train, batch_size=8, shuffle=True)
val_loader = DeviceDataLoader(val, batch_size=8, shuffle=False)
test_loader = DeviceDataLoader(test, batch_size=1, shuffle=False)


loss = WeightedCrossEntropyLoss([1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
model = DeepLabv3(7)
if torch.cuda.is_available():
    model.cuda()

fit(
    30,
    torch.optim.Adam,
    model,
    loss,
    0.00005,
    train_loader,
    val_loader,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
)

best_model = torch.load("saved_models/DeepLabv3_7.pt")
best_model.eval()


def calculate_accuracy(model, test_loader):
    """Calculate the accuracy of the model on the test set."""
    correct_pixels = 0
    total_pixels = 0

    for idx, images, labels in test_loader:
        with torch.no_grad():
            predicted_masks = model(images)
            predicted_masks = torch.argmax(predicted_masks, dim=1)
            correct_pixels += torch.sum(predicted_masks == labels).item()
            total_pixels += torch.numel(labels)

    return correct_pixels / total_pixels


accuracy = calculate_accuracy(best_model, test_loader)
print(f"Test Accuracy: {accuracy}")

from utils import remove_small_artifacts

inverse_color_map = {
    0: np.array((0, 0, 0)),  # background
    1: np.array((128, 0, 0)),  # skin
    2: np.array((0, 128, 0)),  # hair
    3: np.array((128, 128, 0)),  # tshirt
    4: np.array((0, 0, 128)),  # shoes
    5: np.array((128, 0, 128)),  # pants
    6: np.array((0, 128, 128)),  # dress
}

# Plot some of the samples
for idx, image, label in test_loader:
    with torch.no_grad():
        pred_mask = best_model(image)

    # Convert the predicted mask and label to numpy arrays
    pred_mask = pred_mask.argmax(dim=1).cpu().numpy()[0]
    gt_label = label.squeeze(1).cpu().numpy()[0]
    post_process = remove_small_artifacts(pred_mask, min_size=100)

    # Map back the colors to original pallette
    pred_mask_color = np.zeros(
        (pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8
    )
    gt_label_color = np.zeros((gt_label.shape[0], gt_label.shape[1], 3), dtype=np.uint8)
    post_process_color = np.zeros(
        (post_process.shape[0], post_process.shape[1], 3), dtype=np.uint8
    )

    for class_idx, color in inverse_color_map.items():
        pred_mask_color[pred_mask == class_idx] = color
        gt_label_color[gt_label == class_idx] = color
        post_process_color[post_process == class_idx] = color

    # Plot the original image and masks
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(MyPeopleSet.images[idx])
    plt.title(f"Original Image {idx[0] + 1}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_label_color)
    plt.title("Ground Truth Label")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask_color)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(post_process_color)
    plt.title("Post Processed Mask")
    plt.axis("off")
    plt.show()
