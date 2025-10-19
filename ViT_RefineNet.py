import os
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import lightning as L
from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint
from timm.layers import LayerNorm2d
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out
class Encoder(nn.Module):
    def __init__(self, enc_in, enc_out, n_dim, leaky_relu_alpha=0.3):
        super(Encoder, self).__init__()
        self.conv2d = nn.Conv2d(enc_in, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_1 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_2 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_3 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_4 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_5 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_6 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_7 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_8 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.mu = nn.Conv2d(n_dim, enc_out, kernel_size=(3, 3), padding='same')

        self.average_pooling2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.average_pooling2d_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.average_pooling2d_2 = nn.AvgPool2d(kernel_size=(2, 2))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.leaky_relu(self.conv2d(x))
        x = self.leaky_relu(self.conv2d_1(x))
        x = self.leaky_relu(self.conv2d_2(x))
        skip1 = x
        x = self.average_pooling2d(x)
        x = self.leaky_relu(self.conv2d_3(x))
        x = self.leaky_relu(self.conv2d_4(x))
        x = self.leaky_relu(self.conv2d_5(x))
        skip2 = x
        x = self.average_pooling2d_1(x)
        x = self.leaky_relu(self.conv2d_6(x))
        x = self.leaky_relu(self.conv2d_7(x))
        x = self.leaky_relu(self.conv2d_8(x))
        skip3 = x
        x = self.average_pooling2d_2(x)
        x = self.leaky_relu(self.mu(x))
        return x, skip1, skip2, skip3
    


class Decoder(nn.Module):
    def __init__(self, dec_in, dec_out, n_dim, leaky_relu_alpha=0.3, is_gated = 0):
        super(Decoder, self).__init__()
        self.is_gated = is_gated
        self.conv2d_transpose = nn.ConvTranspose2d(dec_in, dec_in, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_1 = nn.ConvTranspose2d(dec_in + n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_4 = nn.ConvTranspose2d(2 * n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_5 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_6 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_7 = nn.ConvTranspose2d(2 * n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_8 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_9 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        if self.is_gated:
            self.att = AttentionBlock(F_g = 4, F_l=27, n_coefficients=128)
            self.att1 = AttentionBlock(F_g=27, F_l=27, n_coefficients=64)
            self.att2 = AttentionBlock(F_g = 27, F_l=27, n_coefficients=32)
        self.up_sampling2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up_sampling2d_1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.up_sampling2d_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.dropout = nn.Dropout()
        self.conv2d_output = nn.Conv2d(n_dim, dec_out, kernel_size=(1, 1))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, skip1, skip2, skip3):
        if self.is_gated:
            x = self.leaky_relu(self.conv2d_transpose(x))
            x = self.up_sampling2d(x)
            s3 = self.att(gate = x, skip_connection = skip3)
            x = torch.cat((x, s3), dim=1)
            x = self.leaky_relu(self.conv2d_transpose_1(x))
            x = self.leaky_relu(self.conv2d_transpose_2(x))
            x = self.leaky_relu(self.conv2d_transpose_3(x))
            x = self.up_sampling2d_1(x)
            s2 = self.att1(gate = x, skip_connection = skip2)
            x = torch.cat((x, s2), dim=1)
            x = self.leaky_relu(self.conv2d_transpose_4(x))
            x = self.leaky_relu(self.conv2d_transpose_5(x))
            x = self.leaky_relu(self.conv2d_transpose_6(x))
            x = self.up_sampling2d_2(x)
            s1 = self.att2(gate = x, skip_connection = skip1)
            x = torch.cat((x, s1), dim=1)
            x = self.leaky_relu(self.conv2d_transpose_7(x))
            x = self.leaky_relu(self.conv2d_transpose_8(x))
            x = self.leaky_relu(self.conv2d_transpose_9(x))
            x = self.conv2d_output(x)
        else:
            x = self.leaky_relu(self.conv2d_transpose(x))
            x = self.up_sampling2d(x)
            x = torch.cat((x, skip3), dim=1)
            x = self.leaky_relu(self.conv2d_transpose_1(x))
            x = self.leaky_relu(self.conv2d_transpose_2(x))
            x = self.leaky_relu(self.conv2d_transpose_3(x))
            x = self.up_sampling2d_1(x)
            x = torch.cat((x, skip2), dim=1)
            x = self.leaky_relu(self.conv2d_transpose_4(x))
            x = self.leaky_relu(self.conv2d_transpose_5(x))
            x = self.leaky_relu(self.conv2d_transpose_6(x))
            x = self.up_sampling2d_2(x)
            x = torch.cat((x, skip1), dim=1)
            x = self.leaky_relu(self.conv2d_transpose_7(x))
            x = self.leaky_relu(self.conv2d_transpose_8(x))
            x = self.leaky_relu(self.conv2d_transpose_9(x))
            x = self.conv2d_output(x)
        return x

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {filename}")
    return model, optimizer

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

output_file = "output_results_vitloc.txt"
def fprint(message):
    with open(output_file, "a") as file:
        file.write(message + "\n")
class SkipAutoencoder(Autoencoder):
    def __init__(self, outpath, enc_in=2, enc_out=4, dec_out=1, n_dim=4, leaky_relu=0.3, is_gate = 0):
        super().__init__()
        self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        self.outpath = outpath
        self.encoder = Encoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu)
        self.decoder = Decoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu, is_gated= is_gate)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class VisionTransformer(L.LightningModule):
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 in_chans=2, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm):
        super(VisionTransformer, self).__init__()
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(img_size, 
                                      patch_size, 
                                      in_chans, 
                                      embed_dim)       
        self.encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=True, 
                  norm_layer=norm_layer
            ) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pred = nn.Linear(embed_dim, patch_size ** 2, bias=True)
        self.skipnet = SkipAutoencoder(enc_in=2,
                        enc_out=4,
                        dec_out=1,
                        n_dim=27,
                        leaky_relu=0.3,
                        outpath="", is_gate = 1)
        self.skip3 = SkipAutoencoder(enc_in=2,
                        enc_out=4,
                        dec_out=1,
                        n_dim=27,
                        leaky_relu=0.3,
                        outpath="", is_gate = 1)
        self.initialize_weights()
        self.save_hyperparameters()
    def initialize_weights(self):
        patch_embed_weights = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_embed_weights.view([
            patch_embed_weights.shape[0], -1]))      
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 
    def forward(self, y):#skip1-vit-2
        x = y.clone()
        y = self.skipnet(y)
        y = y.squeeze(1)
        x[:, 0] = y
        b = x.clone()
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.encoder_blocks(x)
        x = self.norm(x)
        x = self.pred(x)
        c = self.unpatchify(x, 8)
        building = b[:, 1].unsqueeze(1)
        y = y.unsqueeze(1)
        input = torch.cat((c, building), dim = 1)
        return self.skip3(input)       

    def pos_embed(self, x):
        return x + build_sincos2d_pos_embed(self.patch_embed.grid_size, 
                                            dim=x.size(-1),
                                            device=x.device)          
    def unpatchify(self, x, patch_size=16):
        # x: (batch, num_patches, patch_dim)
        # patch_dim = patch_size * patch_size * channels
        b, n, p_dim = x.shape
        h = w = int((n) ** 0.5)
        c = p_dim // (patch_size * patch_size)
        x = x.reshape(b, h, w, patch_size, patch_size, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (b, c, h, ph, w, pw)
        return x.reshape(b, c, h * patch_size, w * patch_size)
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        preds = self.forward(inputs)
        loss = nn.functional.mse_loss(preds.squeeze(), targets)
        self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        preds = self.forward(inputs).squeeze()
        loss = nn.functional.mse_loss(preds, targets)
        self.log('val_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        preds = self.forward(inputs).squeeze()
        loss = nn.functional.mse_loss(preds, targets)
        self.log('test_loss', torch.sqrt(loss), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([            
            {'params': self.skipnet.parameters(), 'lr': 5e-4},  
            {'params': self.skip3.parameters(), 'lr': 5e-4},  
    {'params': [param for name, param in self.named_parameters() if "skipnet" not in name and "skip3" not in name and "skip" not in name], 'lr': 0.001}
])   
        return optimizer
    def save_model(self, optimizer, filename):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
            filename)
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
class DirectionalMapDataset(Dataset):
    def __init__(self, data_path, datatype = 'train'):
        self.data_path = data_path
        self.datatype = datatype
        self.list_images = [f for f in os.listdir(os.path.join(self.data_path, self.datatype , 'masks')) 
                            if f.endswith('.png')]

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        # 获取信号文件名
        mask_file = self.list_images[idx]
        # 构建信号文件路径
        mask_path = os.path.join(self.data_path, self.datatype , 'masks', mask_file)
        building_file = f"{mask_file.split('_')[0]}.png"
        building_path = os.path.join(self.data_path, self.datatype, 'buildings', building_file)
        signal_file = f"{mask_file.split('_')[0]}_{mask_file.split('_')[1]}.png"
        signal_path = os.path.join(self.data_path, self.datatype , 'dpm', signal_file)
        single_signal_image =  np.array(Image.open(signal_path)).astype(np.float32)
        building_image = np.array(Image.open(building_path)).astype(np.float32)
        mask_img = np.array(Image.open(mask_path)).astype(np.float32)
        building_image /= 255.0
        mask_img /= 255.0
        single_signal_image /= 255.0
        masked_signal_image = single_signal_image * mask_img*(1 - building_image)
        sampling_pixel_img = mask_img - building_image
        data = np.stack([masked_signal_image, sampling_pixel_img], axis=0)
        target = single_signal_image *(1 - building_image)
        return data, target

if __name__ == "__main__":
    data_path = './Directional_Dataset'
    model_vit = create_vit_model(img_size=256, patch_size=8,
                        embed_dim=128,
                        depth=6,
                        num_heads=16)
    model_vit=convert_ln_to_dyt(model_vit)
    #print(model_vit)
    random.seed(42)
    torch.manual_seed(0)
    np.random.seed(0)
    train_dataset = DirectionalMapDataset(data_path, datatype='train')
    val_dataset = DirectionalMapDataset(data_path, datatype='val')
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="vitrefinenet-epoch{epoch:03d}-lr{lr:.1e}-val_loss{val_loss:.5f}")
    trainer = L.Trainer(max_epochs=100, accelerator='cuda', callbacks=[checkpoint])
    trainer.fit(model_vit, train_dataloaders=train_loader, val_dataloaders=val_loader)