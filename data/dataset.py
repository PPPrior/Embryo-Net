from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import *
from torch.utils.data import Dataset
from .transforms import Stack, ToTorchFormatTensor, GroupNormalize


class EmbryoDataset(Dataset):
    def __init__(self, image, label, transform=None):
        self.image, self.label = image, label
        self.transform = transform
        assert len(self.image) == len(self.label)

    def __getitem__(self, index):
        if isinstance(self.image[index], tuple):
            return self._getitems(index), self.label[index]
        img = Image.open(self.image[index])

        # TODO: File name on Image
        # draw = ImageDraw.Draw(img)
        # ft = ImageFont.truetype('SimHei', 40)
        # draw.text((1, 1), self.image[index].split('/')[-1], fill='red', font=ft)

        if not self.transform:
            self.transform = Compose([
                ToTensor(),
            ])
        img = self.transform(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.image)

    def _getitems(self, index):
        imgs = [Image.open(i) for i in self.image[index]]
        if not self.transform:
            self.transform = Compose([
                Stack(),
                ToTorchFormatTensor(),
            ])
        imgs = self.transform(imgs)
        return imgs
