from PIL import Image
from torchvision import transforms
from .transforms_custom import *

import numpy as np

def augment(image):
    distortion_perspective = np.random.uniform(0,0.3)

    elastic_dist_magnitude = np.random.randint(1, 20 + 1)
    elastic_dist_kernel = np.random.randint(1, 3 + 1)
    magnitude_w, magnitude_h = (elastic_dist_magnitude, 1) if np.random.randint(2) == 0 else (1, elastic_dist_magnitude)
    kernel_h = np.random.randint(1, 3 + 1)
    kernel_w = np.random.randint(1, 3 + 1)

    br_factor = np.random.uniform(0.01, 1)
    ctr_factor = np.random.uniform(0.01, 1)

    dilation_erosion = None
    if np.random.randint(2) == 0:
        dilation_erosion = Erosion((kernel_w, kernel_h), 1)
    else:
        dilation_erosion = Dilation((kernel_w, kernel_h),1)

    transform = transforms.Compose(
        [   
            transforms.ToPILImage(),
            transforms.RandomPerspective(distortion_scale=distortion_perspective, p=1, interpolation=Image.BILINEAR, fill=255),
            transforms.RandomApply([ElasticDistortion(grid=(elastic_dist_kernel, elastic_dist_kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1,1))], p=0.2),
            transforms.RandomApply([RandomTransform(16)], p=0.2),
            transforms.RandomApply([dilation_erosion], p=0.2),
            transforms.RandomApply([BrighnessAjust(br_factor)], p=0.2),
            transforms.RandomApply([ContrastAdjust(ctr_factor)], p=0.2),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    )
    
    image = transform(image)

    return image

def convert_img_to_tensor(image, force_one_channel=False):
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor()]
    )

    image = transform(image)

    return image