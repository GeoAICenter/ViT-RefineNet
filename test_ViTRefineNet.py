import math
from collections import defaultdict
import sys
#from UNET_PORTING import SkipAutoencoder
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
os.environ["KERAS_BACKEND"] = "torch"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import keras
import random
import matplotlib.pyplot as plt
import lightning as L
# from torch.utils.tensorboard import SummaryWriter
from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from functools import partial
# from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from ViT_RefineNet import VisionTransformer
from ViT_RefineNet import DynamicTanh
from ViT_RefineNet import create_vit_model
from ViT_RefineNet import convert_ln_to_dyt
from functools import partial

custom_norm = partial(DynamicTanh, channels_last=True)
output_file = "output_results_vitloc.txt"
def fprint(message):
    with open(output_file, "a") as file:
        file.write(message + "\n")
def create_vit_model(img_size, 
                     patch_size, 
                     embed_dim, 
                     depth,
                     num_heads):
    model = VisionTransformer(img_size=img_size, 
                              patch_size=patch_size, 
                              embed_dim=embed_dim,
                              depth=depth,
                              num_heads=num_heads)
    return model

def load_vit_model(checkpoint_path,
                   img_size, 
                   patch_size, 
                   embed_dim, 
                   depth,
                   num_heads):
    model = VisionTransformer.load_from_checkpoint(checkpoint_path,
                                                   img_size=img_size,
                                                   patch_size=patch_size,
                                                   embed_dim=embed_dim,
                                                   depth=depth,
                                                   num_heads=num_heads)
    return model
class testDataset(Dataset):
    def __init__(self, data_path, datatype = 'test', general = True, stratified = False, positive_threshold = None):

        self.data_path = data_path
        self.datatype = datatype
        self.stratified = int(stratified)
        self.positive_threshold = positive_threshold
        #self.list_images = [f for f in os.listdir(os.path.join(self.data_path, self.datatype , f'masks_{self.positive_threshold}')) if f.endswith('.png')]
        self.list_images = [f for f in os.listdir(os.path.join(self.data_path, self.datatype , f'masks_smpl_{self.positive_threshold}_random_pos_rate')) if f.endswith('.png')]
        print(f"testlen={len(self.list_images)}")
    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        mask_file = self.list_images[idx]
        #mask_path = os.path.join(self.data_path, self.datatype , f'masks_{self.positive_threshold}', mask_file)
        mask_path = os.path.join(self.data_path, self.datatype , f'masks_smpl_{self.positive_threshold}_random_pos_rate', mask_file)
        building_file = f"{mask_file.split('_')[0]}.png"
        building_path = os.path.join(self.data_path, self.datatype, 'buildings', building_file)
        signal_file = f"{mask_file.split('_')[0]}_{mask_file.split('_')[1]}.png"
        signal_path = os.path.join(self.data_path, self.datatype , 'dpm', signal_file)
        att_path = os.path.join(self.data_path, self.datatype , 'antennas', signal_file)      
        single_signal_image = np.array(Image.open(signal_path)).astype(np.float32)
        building_image = np.array(Image.open(building_path)).astype(np.float32)
        mask_img = np.array(Image.open(mask_path)).astype(np.float32)
        att_img = np.array(Image.open(att_path)).astype(np.float32)
        att_img/=255.0
        single_signal_image /= 255.0
        building_image /= 255.0
        mask_img /= 255.0
        masked_signal_image = single_signal_image * mask_img*(1 - building_image)
        sampling_pixel_img = mask_img - building_image
        data = np.stack([masked_signal_image, sampling_pixel_img], axis=0)
        target = single_signal_image *(1 - building_image)
        total_sample = np.sum(mask_img)
        positive_sample = np.sum(masked_signal_image!=0)
        return data, target, positive_sample/total_sample, total_sample, (1 - building_image), att_img
def loss_fn(pred, target, mask):
    # (B, 1, H, W)
    pred = pred * mask
    target = target * mask
    # Get find the total pixels of propagation field (B, 1, H, W) => (B, 1) each H,W will have x amount of propagation field pixels. 
    all_non_building_pixels = torch.sum(mask.view(-1, 256*256).contiguous(), dim=1).view(-1, 1).contiguous()
    #print(all_non_building_pixels)
    # Still (B, 1, H, W)
    sq_loss = (pred-target)**2
    # Flatten the 2d array into 1d array (B, 1, H, W) => (B, H*W) => (B, 1) after divide to get mean and torch.sum and sqrt to get each map RMSE
    sq_loss = torch.sqrt(torch.sum(sq_loss.view(-1, 256*256).contiguous(), dim=1, keepdim=True)/all_non_building_pixels)
    #print(sq_loss.shape)
    # Get total sum of all RMSE of all maps
    return torch.sum(sq_loss)
def run_test(model_path, data_path, loss_fn, p_id):
    device = f'cuda:{p_id//10 % 5}'
    model_vit = VisionTransformer(img_size=256, patch_size=8,
                     embed_dim=128,
                     depth=6,
                     num_heads=16)
    convert_ln_to_dyt(model_vit)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_vit.load_state_dict(checkpoint['state_dict'])
    model_vit.to(device)
    model_vit.eval()
    test_data = testDataset(data_path, general=False, stratified=True, positive_threshold=p_id)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    test_loss = 0
    test_att = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets, p, t, masks, atts = data
            inputs, targets, masks = inputs.to(device), targets.squeeze().to(device), masks.to(device)
            outputs= model_vit.forward(inputs)
            loss_ = loss_fn(outputs.squeeze(), targets, masks)
            #print(loss_.item()/32)
            test_loss += loss_.mean().item()
            #test_att += r.sum().item()
    avg_loss = test_loss / len(test_data)
    avg_att = test_att / len(test_data)
    print(f'pid={p_id}, avg_loss = {avg_loss}, avg_att = {avg_att}')
    #print(f'pid={p_id}, avg_loss = {avg_loss}')
    '''device = f'cuda:{p_id//10 % 5}'
    model_vit = VisionTransformer(img_size=256, patch_size=8,
                     embed_dim=128,
                     depth=6,
                     num_heads=16)
    #model_vit = VisionTransformer.load_from_checkpoint(model_path)
    convert_ln_to_dyt(model_vit)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_vit.load_state_dict(checkpoint['state_dict'])  # 如果是 PyTorch Lightning
    # model_vit.load_state_dict(checkpoint)  # 如果是纯 PyTorch
    model_vit.to(device)
    model_vit.eval()
    test_att = 0
    test_data = testDataset(data_path, general=False, stratified=True, positive_threshold=p_id)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    total_count = 0
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets, p, t, masks, atts = data
            inputs, targets, masks, atts = inputs.to(device), targets.squeeze().to(device), masks.to(device), atts.to(device)
            outputs = model_vit.forward(inputs)#_, _, outputs, _, r = model_vit.forward(inputs)
            preds = torch.sigmoid(outputs).squeeze()
            #test_att += r.sum().item()
            B, H, W = preds.shape
            flat_idx = torch.argmax(preds.view(B, -1), dim=1)  # [B]
            pred_row = flat_idx // W
            pred_col = flat_idx % W
            pred_coords = torch.stack([pred_row, pred_col], dim=1)  # [B, 2]
            B, H, W = atts.shape
            flat_idx = torch.argmax(atts.view(B, -1), dim=1)  # [B]
            target_row = flat_idx // W
            target_col = flat_idx % W
            target_coords = torch.stack([target_row, target_col], dim=1)  # [B, 2]
            euclidean_dist = torch.sqrt(torch.sum((pred_coords - target_coords) ** 2, dim=1))
            #print(euclidean_dist.sum()/euclidean_dist.numel())
            test_loss += euclidean_dist.sum()
            total_count += euclidean_dist.numel()

    avg_loss = test_loss / len(test_data)
    avg_att = test_att / len(test_data)
    #print(f'pid={p_id}, avg_loss = {avg_loss}, avg_att
    print(f'pid={p_id}, avg_loss = {avg_loss}, avg_att = {avg_att}')
    #print(f'pid={p_id}, avg_loss = {avg_loss}')'''
if __name__ == '__main__':
    p_id = int(sys.argv[1])
    data_path = './Directional_Dataset'
    model_path = '/home/UNT/yl0768/fyb4d14_DATA/version_153/checkpoints/vitrefinet_8x83-epochepoch=029-lrlr=0.0e+00-val_lossval_loss=0.06851.ckpt'
    run_test(model_path, data_path, loss_fn, p_id)
