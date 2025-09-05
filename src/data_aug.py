import torchvision.transforms as T


class DataAugmentation:
    def __init__(self, img_size, transform = None):
        self.img_size = img_size
        self.transform = self._random_compose(transform)

    def _random_compose(self, transform):
        # If a specific transform is provided, use it
        if transform is not None:
            return transform

        return T.Compose([
            T.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.0)),
            T.RandomChoice([T.RandomRotation(degrees=90), 
                            T.RandomVerticalFlip()],
                            T.RandomHorizontalFlip(), p=[0.4, 0.3, 0.3]),
            T.RandomChoice([T.ColorJitter(0.8, 0.8, 0.8, 0.2), 
                            T.RandomGrayscale(p=0.6)], p=[0.5, 0.5]),
            T.RandomApply([T.GaussianBlur(kernel_size=min(21, int(0.1 * self.img_size)*2+1), sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor()])

    def __call__(self, x):
        return self.transform(x), self.transform(x)