from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
import os


class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __iter__(self):
        for batch in super().__iter__():
            if isinstance(batch, (list, tuple)):
                yield tuple(b.to(self.device) for b in batch)
            else:
                yield batch.to(self.device)


class ImageSet(Dataset):
    """
    A custom dataset class for loading images and labels.

    Args:
        image_dir (str): The directory path containing the images.
        label_dir (str): The directory path containing the labels.
        normalize (bool, optional): Flag to normalize the image tensors. Defaults to True.
        transform (torchvision.transforms, optional): A torchvision transform to apply to the images. Defaults to None.
    """

    def __init__(
        self,
        image_dir,
        label_dir,
        normalize=True,
        transform=None,
        label_type="binary",
        use_cache=True,
    ):
        # Load images
        img_names = np.array(sorted(os.listdir(image_dir)))
        print(f"Loading Images: {img_names[0]}, ..., {img_names[-1]}")

        img_paths = [os.path.join(image_dir, n) for n in img_names]
        self.images = [cv2.imread(p) for p in img_paths]
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images]

        tensors = torch.stack(
            [torch.from_numpy(img).float() / 255 for img in self.images]
        )
        if normalize:
            if use_cache:
                self.mean = torch.tensor([0.5875, 0.5572, 0.5204])
                self.std = torch.tensor([0.2818, 0.2785, 0.2905])
            else:
                self.mean = torch.mean(tensors, dim=(0, 1, 2))
                self.std = torch.std(tensors, dim=(0, 1, 2))
            tensors = (tensors - self.mean) / self.std

        self.image_tensors = tensors.permute(0, 3, 1, 2)

        # Load labels
        label_names = np.array(sorted(os.listdir(label_dir)))
        print(f"Loading Labels: {label_names[0]}, ..., {label_names[-1]}")

        label_paths = [os.path.join(label_dir, n) for n in label_names]

        if label_type == "binary":
            self.labels = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in label_paths]
            for label in self.labels:
                label[label > 0] = 1
        else:
            self.labels = []
            for p in label_paths:
                image = cv2.imread(p)
                color_map = {
                    (0, 0, 0): 0,  # background
                    (0, 0, 128): 1,  # skin
                    (0, 128, 0): 2,  # hair
                    (0, 128, 128): 3,  # tshirt
                    (128, 0, 0): 4,  # shoes
                    (128, 0, 128): 5,  # pants
                    (128, 128, 0): 6,  # dress
                }

                label_seg = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                for color, label in color_map.items():
                    color = np.asarray(color)
                    label_seg[(image == color).all(axis=2)] = label

                self.labels.append(label_seg)

        self.label_tensors = torch.stack(
            [torch.from_numpy(label) for label in self.labels]
        )

        self.label_tensors = self.label_tensors.unsqueeze(1)
        self.transform = transform

    def __getitem__(self, index):
        image = self.image_tensors[index]
        label = self.label_tensors[index]

        if self.transform is not None:
            image, label = self.transform(image, label)
        return index, image, label

    def __len__(self):
        return len(self.images)
