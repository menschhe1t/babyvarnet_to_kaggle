import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return  A.Compose([
                                A.Resize(800, 800),
                                # A.HorizontalFlip(p=0.5),
                                # A.ColorJitter(0.4, 0.4, 0.4, 0.4, p=0.5),
                                # A.GaussNoise(var_limit=5. / 255., p=0.3),
                                # A.Normalize(mean=(0.3, 0.3, 0.3), std=(0.3, 0.3, 0.3), always_apply=False, p=1.0),
                                ToTensorV2()],        p=1.0, 
    )

def get_valid_transform():
    return  A.Compose([
                                A.Resize(800, 800),
                                # A.HorizontalFlip(p=0.5),
                                # A.ColorJitter(0.4, 0.4, 0.4, 0.4, p=0.5),
                                # A.GaussNoise(var_limit=5. / 255., p=0.3),
                                # A.Normalize(mean=(0.3, 0.3, 0.3), std=(0.3, 0.3, 0.3), always_apply=False, p=1.0),
                                ToTensorV2()],        p=1.0, 
    )

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
        #self.transform = get_train_transform
    def __call__(self, input, target, attrs, fname, slice):
        print(input.shape)
        print(target.shape)

        input = cv2.resize(input[:,:, np.newaxis], (800,800))
            
        # input = to_tensor(input)
        if not self.isforward:
            # target = to_tensor(target)
            target = cv2.resize(target[:,:, np.newaxis], (800,800))
            target = torch.squeeze(torch.tensor(target))
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        input = torch.squeeze(torch.tensor(input))
        return input, target, maximum, fname, slice
