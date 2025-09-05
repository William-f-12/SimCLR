import random
import torchvision.transforms as T


class DataAugmentation:
    def __init__(self, img_size, transform = None):
        self.img_size = img_size
        self.color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = self._random_compose(transform)

    def _random_compose(self, transform):
        # If a specific transform is provided, use it
        if transform is not None:
            return transform

        if random.random() < 0.7:
            return T.Compose([
                T.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.0)),
                T.RandomRotation(degrees = 180),
                T.RandomApply([self.color_jitter], p=0.8),
                T.GaussianBlur(kernel_size=int(0.1 * self.img_size) * 2 + 1, sigma=(0.1, 2.0)),
                T.ToTensor()
                ])
        else:
            return T.Compose([
                T.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=int(0.1 * self.img_size) * 2 + 1, sigma=(0.1, 2.0)),
                T.ToTensor()
                ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)