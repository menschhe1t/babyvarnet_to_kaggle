import numpy as np
import torch
import cv2

from tqdm import tqdm
from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.unet import Unet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)
    loop = tqdm(data_loader)
    with torch.no_grad():
        for itr, data in enumerate(loop):
            input, _, _, fnames, slices = data[0]
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                img_size = 384
                
                input_i = input[i].cpu().numpy()
                reconstruction_i =  output[i].cpu().numpy()
               
                input_i = np.squeeze(cv2.resize(input_i[:,:, np.newaxis], (img_size,img_size)))
                output_i = np.squeeze(cv2.resize(reconstruction_i[:,:, np.newaxis], (img_size,img_size)))
                
                reconstructions[fnames[i]][int(slices[i])] = reconstruction_i
                inputs[fnames[i]][int(slices[i])] = input_i

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    return reconstructions, inputs


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, mode='test', args = args, data_type = 'input', isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)

    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
