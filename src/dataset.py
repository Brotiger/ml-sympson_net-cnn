import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

RESCALE_SIZE = 224
DATA_MODES = ['train', 'val', 'test']

class SimpsonDataset(Dataset):
    def __init__(self, files, label_encoder, mode):
        super().__init__()
        self.files = files
        self.label_encoder = label_encoder
        self.mode = mode

        if self.mode == 'test':
            self.files = sorted(files, key=self.extract_image_number)

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]

        self.len_ = len(self.files)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        return image

    def extract_image_number(self, path):
        filename = os.path.basename(path)
        return int(filename[3:-4])

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)