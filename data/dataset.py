from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import *
from torch.utils.data import Dataset


class EmbryoDataset(Dataset):
    def __init__(self, image, label, transform=None):
        self.image, self.label = image, label
        self.transform = transform
        assert len(self.image) == len(self.label)

    def __getitem__(self, index):
        img = Image.open(self.image[index])

        # TODO: File name on Image
        # draw = ImageDraw.Draw(img)
        # ft = ImageFont.truetype('SimHei', 40)
        # draw.text((1, 1), self.image[index].split('/')[-1], fill='red', font=ft)

        if self.transform:
            img = self.transform(img)
        else:
            transform = Compose([
                ToTensor(),
            ])
            img = transform(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.image)
