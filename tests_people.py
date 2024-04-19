import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ImageDataset import ImageSet, DeviceDataLoader
from model import DeepLabv3
from trainers import fit
from loss import CrossEntropyLoss

image_dir = "data/images"
label_dir = "data/labels/person"

MyPeopleSet = ImageSet(image_dir, label_dir)

# Set the seed for the random split
torch.manual_seed(42)

train, val, test = torch.utils.data.random_split(MyPeopleSet, (0.5, 0.1, 0.4))
train_loader = DeviceDataLoader(train, batch_size=8, shuffle=True)
val_loader = DeviceDataLoader(val, batch_size=8, shuffle=False)
test_loader = DeviceDataLoader(test, batch_size=1, shuffle=False)


loss = CrossEntropyLoss()
model = DeepLabv3(2)
if torch.cuda.is_available():
    model.cuda()

# fit(
#     25,
#     torch.optim.Adam,
#     model,
#     loss,
#     0.00005,
#     train_loader,
#     val_loader,
#     torch.optim.lr_scheduler.ReduceLROnPlateau,
# )

best_model = torch.load("saved_models/DeepLabv3_2.pt")
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


import matplotlib.image
from utils import remove_small_artifacts

# Plot some of the samples
for idx, image, label in test_loader:
    with torch.no_grad():
        pred_mask = best_model(image)

    # Convert the predicted mask and label to numpy arrays
    pred_mask = pred_mask.argmax(dim=1).cpu().numpy()[0]
    gt_label = label.squeeze(1).cpu().numpy()[0]
    post_process = remove_small_artifacts(pred_mask)

    # Plot the original image, and masks
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(MyPeopleSet.images[idx])
    plt.title(f"Original Image {idx[0] + 1}")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(gt_label, cmap="gray")
    plt.title("Ground Truth Label")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(post_process, cmap="gray")
    plt.title("Post Processed Mask")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(MyPeopleSet.images[idx])
    plt.imshow(post_process, cmap="jet", alpha=0.5)
    plt.title("Overlayed Mask")
    plt.axis("off")
    plt.show()

    # matplotlib.image.imsave(f'data/output/mask_{(index+1):04d}.png', post_process)
