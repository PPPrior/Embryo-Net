from PIL import Image
from torchvision.transforms import *
from torch.utils.data import Dataset
from .utils import get_data


class EmbryoDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.image, self.label = get_data(data_root)
        self.transform = transform
        assert len(self.image) == len(self.label)

    def __getitem__(self, index):
        img = Image.open(self.image[index])
        if self.transform:
            img = self.transform(img)
        else:
            transform = Compose([
                CenterCrop(448),
                ToTensor(),
            ])
            img = transform(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.image)
