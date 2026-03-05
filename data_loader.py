from torch.utils.data import Dataset
from PIL import Image
import os

class DeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.blur_dir = os.path.join(root_dir, "blur")
        self.sharp_dir = os.path.join(root_dir, "sharp")
        self.blur_images = sorted(os.listdir(self.blur_dir))
        self.sharp_images = sorted(os.listdir(self.sharp_dir))
        self.transform = transform

        assert len(self.blur_images) == len(self.sharp_images)

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.blur_images[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp_images[idx])

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return {"blur": blur_img, "sharp": sharp_img}