"""Image transforms shared by SimCLR pretraining and downstream evaluation."""

import torch
import torch.nn as nn
import kornia.augmentation as K

from PIL import Image
from torchvision import transforms


# Same normalization as the preprocessor of google/vit-base-patch16-224
# (inputs scaled to [-1, 1]). It also keeps the input contract identical for
# both stages.
IMAGE_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class SimCLRTransform:
    """CPU (DataLoader worker) part of the SimCLR augmentation.

    Only the spatial augmentations run here, independently for each of the
    two views: they depend on each image's original size and bring every
    view to the same shape, so that the views can be batched. Views are
    returned as uint8 tensors; the photometric augmentations and the
    normalization run afterwards, on the GPU, in `SimCLRGPUAugmentation`.
    """

    def __init__(self, image_size=IMAGE_SIZE):
        self.view_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ])

    def __call__(self, image):
        pil_image = Image.fromarray(image)
        return self.view_transform(pil_image), self.view_transform(pil_image)


class SimCLRGPUAugmentation(nn.Module):
    """GPU part of the SimCLR augmentation, applied to a whole batch of views.

    Random parameters are drawn independently for each view, as in the
    per-image torchvision pipeline it replaces.
    """

    def __init__(self):
        super().__init__()
        self.augment = nn.Sequential(
            K.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=0.5),
        )
        self.normalize = K.Normalize(mean=torch.tensor(MEAN), std=torch.tensor(STD))

    @torch.no_grad()
    def forward(self, views):
        views = views.float() / 255.0
        return self.normalize(self.augment(views))


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
