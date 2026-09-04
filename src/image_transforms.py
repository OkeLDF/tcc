"""Image transforms shared by SimCLR pretraining and downstream evaluation."""

from PIL import Image
from torchvision import transforms


# ImageNet normalization is appropriate for initialization from the public ViT
# checkpoint. It also keeps the input contract identical for both stages.
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SimCLRTransform:
    """Independently generate the two normalized views required by SimCLR."""

    def __init__(self, image_size=IMAGE_SIZE):
        self.view_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __call__(self, image):
        pil_image = Image.fromarray(image)
        return self.view_transform(pil_image), self.view_transform(pil_image)


class EvaluationTransform:
    """Deterministic ViT preprocessing for downstream train and test samples."""

    def __init__(self, image_size=IMAGE_SIZE):
        resize_size = int(image_size / 0.875)
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __call__(self, image):
        return self.transform(Image.fromarray(image))
