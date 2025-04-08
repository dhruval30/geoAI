import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class ImageMaskLoader(Dataset):
    """
    Custom Dataset for loading images and corresponding masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to fetch.

        Returns:
            dict: A dictionary containing the image and its corresponding mask.
        """
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        # Open image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
