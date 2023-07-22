import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import cv2

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet
from tqdm import tqdm

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# def get_train_transform():
#     return  A.Compose([
#                                 A.Resize(800, 800),
#                                 # A.HorizontalFlip(p=0.5),
#                                 # A.ColorJitter(0.4, 0.4, 0.4, 0.4, p=0.5),
#                                 # A.GaussNoise(var_limit=5. / 255., p=0.3),
#                                 # A.Normalize(mean=(0.3, 0.3, 0.3), std=(0.3, 0.3, 0.3), always_apply=False, p=1.0),
#                                 ToTensorV2()],        p=1.0, 
#     )

# def get_valid_transform():
#     return  A.Compose([
#                                 #A.Resize(800, 800),
#                                 # A.HorizontalFlip(p=0.5),
#                                 # A.ColorJitter(0.4, 0.4, 0.4, 0.4, p=0.5),
#                                 # A.GaussNoise(var_limit=5. / 255., p=0.3),
#                                 # A.Normalize(mean=(0.3, 0.3, 0.3), std=(0.3, 0.3, 0.3), always_apply=False, p=1.0),
#                                 ToTensorV2()],        p=1.0, 
#     )

def get_lr_scheduler(mode, optimizer, T):
    if mode == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=T, gamma=0.5)
    elif mode == 'CosineAnnealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T, eta_min=0)
    elif mode == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T, T_mult=2, eta_min=0)


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    loop = tqdm(data_loader)
    for iter, data in enumerate(loop):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(input)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #if iter % args.report_interval == 0:
        loop.set_description(f"Train Epoch [{(epoch+1):3d}/{args.num_epochs:3d}]")
        loop.set_postfix(loss=loss.item()) 
        
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch



def validate(args,epoch, model, data_loader, loss_type):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()
    loop = tqdm(data_loader)
    with torch.no_grad():
        for iter, data in enumerate(loop):
            input, target, maximum, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)
         
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            loss = loss_type(output, target, maximum)
            
            for i in range(output.shape[0]):
                img_size = 384
                output_i =  output[i].cpu().numpy()
                target_i = target[i].cpu().numpy()
                input_i = input[i].cpu().numpy()
                
                input_i = np.squeeze(cv2.resize(input_i[:,:, np.newaxis], (img_size,img_size)))
                target_i = np.squeeze(cv2.resize(target_i[:,:, np.newaxis], (img_size,img_size)))
                output_i = np.squeeze(cv2.resize(output_i[:,:, np.newaxis], (img_size,img_size)))
                
                reconstructions[fnames[i]][int(slices[i])] = output_i
                targets[fnames[i]][int(slices[i])] = target_i
                inputs[fnames[i]][int(slices[i])] = input_i
                
            #if iter % args.report_interval == 0:
            loop.set_description(f"Valid Epoch [{(epoch+1):3d}/{args.num_epochs:3d}]")
            loop.set_postfix(loss=loss.item()) 
            

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)

    
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        # f=exp_dir / 'model.pt'
        f=exp_dir / 'best_model.pt'
    )
    # if is_new_best:
    #     shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    lr_scheduler = get_lr_scheduler('CosineAnnealingWarmRestarts', optimizer, 5)
    
    best_val_loss = 1.
    start_epoch = 0
    
    # train_transform = get_train_transform()
    # valid_transform = get_valid_transform()
    
    input_train_loader = create_data_loaders(data_path = args.data_path_train,mode = 'train' ,args = args, shuffle=True, data_type = 'input')
    grappa_train_loader = create_data_loaders(data_path = args.data_path_train,mode = 'train' ,args = args, shuffle=True, data_type = 'grappa')
    input_val_loader = create_data_loaders(data_path = args.data_path_val,mode = 'valid' ,args = args, data_type = 'input')
    grappa_val_loader = create_data_loaders(data_path = args.data_path_val,mode = 'valid' ,args = args, data_type = 'grappa')
    
    val_loss_log = np.empty((0, 2))
    
    for epoch in range(start_epoch, args.num_epochs):
        # print(f'Epoch #{(epoch+1):2d} ............... {args.net_name} ...............')
        print('input_train')
        train_loss1, train_time1 = train_epoch(args, epoch, model, input_train_loader, optimizer, loss_type)
        print('grappa_train')
        train_loss2, train_time2 = train_epoch(args, epoch, model, grappa_train_loader, optimizer, loss_type)
        print('input_val')
        val_loss1, num_subjects1, reconstructions1, targets1, inputs1, val_time1 = validate(args, epoch, model, input_val_loader, loss_type)
        print('grappa_val')
        val_loss2, num_subjects2, reconstructions2, targets2, inputs2, val_time2 = validate(args, epoch, model, grappa_val_loader, loss_type)

        val_loss = val_loss1 + val_loss2
        reconstructions = np.concatenate((reconstructions1, reconstructions2), axis=0)
        targets = np.concatenate((targets1, targets2), axis=0)
        inputs = np.concatenate((inputs1, inputs2), axis=0)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        
        val_loss = val_loss / (num_subjects1 + num_subjects2)
        is_new_best = val_loss < best_val_loss 
        best_val_loss = min(best_val_loss, val_loss)

        lr_scheduler.step()
        
        if is_new_best == True:
            save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{(epoch+1):4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
        
