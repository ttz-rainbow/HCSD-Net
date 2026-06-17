import os
from PIL import Image as Image
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label, hue):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            hue = F.pad(hue, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            hue = F.pad(hue, (self.size[1] - hue.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            hue = F.pad(hue, (0, self.size[0] - hue.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w), F.crop(hue, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label, hue):
        for t in self.transforms:
            image, label, hue = t(image, label, hue)
        return image, label, hue


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, hue):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label), F.hflip(hue)
        return img, label, hue


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label, hue):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label),F.to_tensor(hue)



def train_dataloader(path, batch_size=1, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(128),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        data_Dataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'valid')
    # transform = PairCompose(
    #     [
    #         PairRandomCrop(384),
    #         PairToTensor()
    #     ]
    # )
    dataloader = DataLoader(
        data_Dataset(image_dir,is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        data_Dataset(os.path.join(path, 'valid')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class data_Dataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'input/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx]))
        h = Image.open(os.path.join(self.image_dir, 'hue', self.image_list[idx]))

        if self.transform:
            image,label,h = self.transform(image, label, h)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            h = F.to_tensor(h)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, h, name
        return image, label, h

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg','tif']:
                raise ValueError
