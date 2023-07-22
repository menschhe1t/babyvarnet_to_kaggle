import numpy as np
import torch
import cv2

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
    def __init__(self, isforward, max_key,mode):
        self.isforward = isforward
        self.max_key = max_key
        self.mode = mode
    def __call__(self, input, target, attrs, fname, slice):
        img_size = 100
        if self.mode == 'train':
            input = cv2.resize(input[:,:, np.newaxis], (img_size,img_size))
            if not self.isforward:
                # target = to_tensor(target)
                target = cv2.resize(target[:,:, np.newaxis], (img_size,img_size))
                target = torch.squeeze(torch.tensor(target))
                maximum = attrs[self.max_key]
            else:
                target = -1
                maximum = -1
            input = torch.squeeze(torch.tensor(input))
            return input, target, maximum, fname, slice
            
        elif self.mode == 'valid':
            input = cv2.resize(input[:,:, np.newaxis], (img_size,img_size))
            if not self.isforward:
                # target = to_tensor(target)
                target = cv2.resize(target[:,:, np.newaxis], (img_size,img_size))
                target = torch.squeeze(torch.tensor(target))
                maximum = attrs[self.max_key]
            else:
                target = -1
                maximum = -1
            input = torch.squeeze(torch.tensor(input))
            return input, target, maximum, fname, slice
        
        elif self.mode == 'test':
            input = cv2.resize(input[:,:, np.newaxis], (img_size,img_size))
            if not self.isforward:
                # target = to_tensor(target)
                target = cv2.resize(target[:,:, np.newaxis], (img_size,img_size))
                target = torch.squeeze(torch.tensor(target))
                maximum = attrs[self.max_key]
            else:
                target = -1
                maximum = -1
            input = torch.squeeze(torch.tensor(input))
            return input, target, maximum, fname, slice
   
            
