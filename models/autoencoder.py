import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
from models.cnnencoder import CNNEncoder
import pdb
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            )
        )
        self.image_size = (200, 200)
        self.x_range = (0, 100)
        self.y_range = (0, 100)
        self.conv = CNNEncoder(in_channels=1, size=200)
        self.trans_layer = nn.Linear(config.encoder_dim + 64, config.encoder_dim)

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z
    
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        ####
        image_map = self.generate_map()
        heat_map = self.generate_heatmap(batch[1], self.image_size, self.x_range, self.y_range)
        map_x_encoded = self.fuse_maps(image_map, heat_map)
        concat_x = torch.cat((map_x_encoded, encoded_x), dim=1)
        encoded_x = self.trans_layer(concat_x)
        ####
        predicted_y_vel = self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def generate_map(self):
        # 定义图像转换
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 读取图像
        image_path = "MAP.png"
        image = Image.open(image_path).convert("L")  # 转换为灰度图
        # 定义调整大小的变换
        resize_transform = transforms.Resize((200, 200))
        image = resize_transform(image)
        image.save("MAP_gray.png")
        image_tensor = transform(image)
        return image_tensor

    def generate_heatmap(self, x_t, image_size, x_range, y_range):
        x_scale = image_size[1] / (x_range[1] - x_range[0])
        y_scale = image_size[0] / (y_range[1] - y_range[0])

        x_pixel = ((x_t[:, :, 0] - x_range[0]) * x_scale).long()
        y_pixel = image_size[0] - ((x_t[:, :, 1] - y_range[0]) * y_scale).long()

        valid_indices = (~torch.isnan(x_t[:, :, 0])) & (~torch.isnan(x_t[:, :, 1]))
        heatmap = torch.zeros(x_t.shape[0], *image_size)

        size = 36
        sigma = 3.0
        kernel = torch.exp(-torch.arange(-size // 2, size // 2 + 1).float() ** 2 / (2 * sigma ** 2))
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()


        for i in range(x_t.shape[1]):
            for b in range(x_t.shape[0]):
                if not valid_indices[b, i]:
                    continue

                x_start = torch.clamp(x_pixel[b, i] - size // 2, 0, image_size[1] - size)
                y_start = torch.clamp(y_pixel[b, i] - size // 2, 0, image_size[0] - size)
                x_end = torch.clamp(x_pixel[b, i] + size // 2, 0, image_size[1])
                y_end = torch.clamp(y_pixel[b, i] + size // 2, 0, image_size[0])

                kernel_size_x = x_end - x_start
                kernel_size_y = y_end - y_start
                adjusted_kernel = kernel[:kernel_size_y, :kernel_size_x]
                heatmap[b, y_start:y_end, x_start:x_end] += adjusted_kernel

        max1, _ = heatmap.max(dim=1, keepdim=True)
        max2, _ = max1.max(dim=2, keepdim=True)
        heatmap = heatmap / (max2 + 1e-10)
        return heatmap

    def fuse_maps(self, image_map, heat_map, alpha=0.7):
        fused_map = alpha * heat_map + (1 - alpha) * image_map
        input = fused_map.unsqueeze(1).cuda()
        map_x_encoded = self.conv(input)
        return map_x_encoded

    # def fusemaps(self, image_map, heat_map, alpha=0.7):
    #     fused_map = alpha * heat_map + (1 - alpha) * image_map
    #     return fused_map

    def get_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        image_map = self.generate_map()
        heat_map = self.generate_heatmap(x_t, self.image_size, self.x_range, self.y_range)
        map_x_encoded = self.fuse_maps(image_map, heat_map)

        feat_x_encoded = self.encode(batch, node_type)  # B * 256

        concat_x = torch.cat((map_x_encoded, feat_x_encoded), dim=1)

        cond = self.trans_layer(concat_x)

        loss = self.diffusion.get_loss(y_t.cuda(), cond)
        return loss
