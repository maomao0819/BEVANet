import os
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNet():
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.transform = self.get_transform()

    def get_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

    def get_dataset(self):
        data_root = os.path.join(self.root, self.split)
        try:
            dataset = ImageFolder(data_root, self.transform)
            return dataset
        except FileNotFoundError as e:
            print(f"Error loading dataset from {data_root}: {e}")
            return None