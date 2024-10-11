import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
# def cva( T1, T2):
#     norm = True
#     if norm:
#         stdT1 = torch.std(T1, dim=[1, 2], keepdim=True)
#         stdT2 = torch.std(T2, dim=[1, 2], keepdim=True)
#         meanT1 = torch.mean(T1, dim=[1, 2], keepdim=True)
#         meanT2 = torch.mean(T2, dim=[1, 2], keepdim=True)
#         # print(T1.shape,meanT1.shape,stdT1.shape)
#         normT1 = (T1 - meanT1) / stdT1
#         normT2 = (T2 - meanT2) / stdT2
#         # normT1=self.norm(T1)
#         # imgT1=torch.var()
#         # img_diff=torch.square(T1-T2)
#
#         img_diff = torch.square(normT1 - normT2)
#         L2_norm = torch.sqrt(img_diff)
#     else:
#         img_diff = torch.square(T1 - T2)
#         L2_norm = torch.sqrt(img_diff)
#     # print(stdT1.shape,L2_norm.shape)
#     return L2_norm

# from util.train_util import DIST
from torch.nn.modules.padding import ReplicationPad2d
# from  models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,SynchronizedBatchNorm2d,resnet101_diff
import math
class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASSP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6,
                               dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12,
                               dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=18,
                               dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.convf = nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1, bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # channels first
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class _PSPModulenobn(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModulenobn, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer)
                                     for pool_size in pool_sizes])
        out_channelsn=in_channels // 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channelsn,
                      kernel_size=3, padding=1, bias=False),
            # norm_layer(out_channelsn),
            # nn.ReLU(),
            # nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # bn = norm_layer(out_channels)
        relu = nn.ReLU()
        return nn.Sequential(prior, conv, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]

        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
class FCSiamDiff(nn.Module):
    def __init__(self, in_dim=3,out_dim=2):
        super(FCSiamDiff, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.ASSP = ASSP(64, 128)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(192, 192, kernel_size=1, padding=0)
        # )
        # self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_1(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_1(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_1(feature_4)
        return down_feature_4, feature_4, feature_3, feature_2, feature_1
    def cva(self,T1,T2):
        norm=True
        if norm:
            stdT1 = torch.std(T1, dim=[1, 2],keepdim=True)
            stdT2 = torch.std(T2, dim=[1, 2],keepdim=True)
            meanT1 = torch.mean(T1,dim=[1,2],keepdim=True)
            meanT2 = torch.mean(T2,dim=[1,2],keepdim=True)
            # print(T1.shape,meanT1.shape,stdT1.shape)
            normT1 = (T1-meanT1)/stdT1
            normT2 = (T2 - meanT2) / stdT2
            # normT1=self.norm(T1)
            # imgT1=torch.var()
            # img_diff=torch.square(T1-T2)

            img_diff=torch.square(normT1-normT2)
            L2_norm=torch.sqrt(img_diff)
        else:
            img_diff=torch.square(T1-T2)
            L2_norm = torch.sqrt(img_diff)
        # print(stdT1.shape,L2_norm.shape)
        return L2_norm

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        # cdpre=self.cva(pre_data[:, 3, :, :],post_data[:, 3, :, :])
        pre_data = pre_data[:, 0:3, :, :]
        post_data = post_data[:, 0:3, :, :]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)
        out=torch.cat([down_feature_41,down_feature_42],1)
        # out=torch.abs(down_feature_41 - down_feature_42)

        up_feature_5 = self.up_sample_1(out)
        # print('up_feature_5',up_feature_5.shape,torch.abs(feature_41 - feature_42).shape)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        DA = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([DA, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        # output = F.softmax(output_feature, dim=1)
        # diffout= torch.mean(torch.square(feature_41 - feature_42),dim=1)
        # diffout=torch.mean(torch.abs(feature_41 - feature_42),dim=1)
        # diffout = F.sigmoid(diffout)
        # print('diffout',diffout.shape)
        # output_featurePatch=self.avgpool(output)
        return output_feature,[DA, feature_42],concat_feature_8


class FCSiamDiffMaskAttention(nn.Module):
    def __init__(self, in_dim=3,out_dim=2):
        super(FCSiamDiffMaskAttention, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.ASSP = ASSP(64, 128)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(192, 192, kernel_size=1, padding=0)
        # )
        # self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )

        self.query_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.key_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.query_convchg = nn.Conv2d(32, 32, kernel_size=1)
        self.key_convchg = nn.Conv2d(32, 32, kernel_size=1)

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_1(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_1(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_1(feature_4)
        return down_feature_4, feature_4, feature_3, feature_2, feature_1
    def cva(self,T1,T2):
        norm=True
        if norm:
            stdT1 = torch.std(T1, dim=[1, 2],keepdim=True)
            stdT2 = torch.std(T2, dim=[1, 2],keepdim=True)
            meanT1 = torch.mean(T1,dim=[1,2],keepdim=True)
            meanT2 = torch.mean(T2,dim=[1,2],keepdim=True)
            # print(T1.shape,meanT1.shape,stdT1.shape)
            normT1 = (T1-meanT1)/stdT1
            normT2 = (T2 - meanT2) / stdT2
            # normT1=self.norm(T1)
            # imgT1=torch.var()
            # img_diff=torch.square(T1-T2)

            img_diff=torch.square(normT1-normT2)
            L2_norm=torch.sqrt(img_diff)
        else:
            img_diff=torch.square(T1-T2)
            L2_norm = torch.sqrt(img_diff)
        # print(stdT1.shape,L2_norm.shape)
        return L2_norm

    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        # cdpre=self.cva(pre_data[:, 3, :, :],post_data[:, 3, :, :])
        pre_data = pre_data[:, 0:3, :, :]
        post_data = post_data[:, 0:3, :, :]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)
        out=torch.cat([down_feature_41,down_feature_42],1)
        # out=torch.abs(down_feature_41 - down_feature_42)

        up_feature_5 = self.up_sample_1(out)
        # print('up_feature_5',up_feature_5.shape,torch.abs(feature_41 - feature_42).shape)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        DA = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([DA, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        mask = torch.argmax(output_feature, dim=1).detach().unsqueeze(1)

        # print('concat_feature_8',concat_feature_8.shape,mask.shape)
        masked_featuresunchg = concat_feature_8 * (1-mask)#[14, 32, 256, 256])
        # print('masked_features',masked_features.shape)
        queryunchg = torch.sum(masked_featuresunchg, dim=(2, 3), keepdim=True) / ((1-mask).sum(dim=(2, 3), keepdim=True) + 1e-5)
        queryunchg = self.query_conv(queryunchg)#torch.Size([14, 32, 1, 1])
        queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)

        # Step 2: Compute similarity between the query and all locations in the feature map
        keyunchg = self.key_conv(masked_featuresunchg)#[14, 32, 256, 256])
        keyunchg = keyunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)

        similarityunchg = torch.bmm(queryunchg, keyunchg)  # Batch matrix multiplication
        similarityunchg = similarityunchg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2), concat_feature_8.size(3))
        attn_weightsunchg = F.softmax(similarityunchg, dim=1)#[14, 14, 256, 256])

        # similarity = F.conv2d(key, query)#[14, 14, 256, 256])

        masked_featureschg = concat_feature_8 * mask  # [14, 32, 256, 256])
        # print('masked_features',masked_features.shape)
        querychg = torch.sum(masked_featureschg, dim=(2, 3), keepdim=True) / (
                    mask.sum(dim=(2, 3), keepdim=True) + 1e-5)
        querychg = self.query_convchg(querychg)  # torch.Size([14, 32, 1, 1])
        querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)

        # Step 2: Compute similarity between the query and all locations in the feature map
        keychg = self.key_convchg(masked_featureschg)  # [14, 32, 256, 256])
        keychg = keychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)

        similaritychg = torch.bmm(querychg, keychg)  # Batch matrix multiplication
        similaritychg = similaritychg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))


        # Step 3: Apply similarity weights to feature map
        attn_weightschg = F.softmax(similaritychg, dim=1)#[14, 14, 256, 256])
        # print('similarity',masked_features.shape,attn_weights.shape,similarity.shape,key.shape,query.shape)

        attn_features = masked_featuresunchg * attn_weightsunchg+masked_featureschg*attn_weightschg
        output_feature = self.conv_block_8(attn_features)

        return output_feature,[feature_41, feature_42,out],attn_features

class FCSiamDiffMaskAttentionCenter(nn.Module):
    def __init__(self, in_dim=3,out_dim=2,unchgN=5,chgN=5):
        super(FCSiamDiffMaskAttentionCenter, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.unchgN=unchgN
        self.chgN=chgN
        self.query_convunchg = nn.Conv2d(32*self.unchgN, 32, kernel_size=1)
        self.key_convunchg = nn.Conv2d(32, 32, kernel_size=1)
        self.query_convchg = nn.Conv2d(32*self.chgN, 32, kernel_size=1)
        self.key_convchg = nn.Conv2d(32, 32, kernel_size=1)

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_1(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_1(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_1(feature_4)
        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def cva(self,T1,T2):
        difference = T2[:, 0:3, :, :] - T1[:, 0:3, :, :]
        magnitude = torch.norm(difference, dim=1)
        # direction = torch.atan2(difference[1], difference[0])
        change_map = (magnitude > 0.8).float()
        # print('change_map',change_map.shape)
        return change_map

    def forward(self, pre_data, post_data,DomainLabel=0,Scenter=None):
        #####################
        # decoder
        #####################
        # cdpre=self.cva(pre_data,post_data)
        # pre_data = pre_data[:, 0:3, :, :]
        # post_data = post_data[:, 0:3, :, :]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data[:, 0:3, :, :])
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data[:, 0:3, :, :])
        out=torch.cat([down_feature_41,down_feature_42],1)
        # out=torch.abs(down_feature_41 - down_feature_42)

        up_feature_5 = self.up_sample_1(out)
        # print('up_feature_5',up_feature_5.shape,torch.abs(feature_41 - feature_42).shape)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        DA = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([DA, torch.abs(feature_11 - feature_12)], dim=1)
        # with torch.no_grad():
        output_feature = self.conv_block_8(concat_feature_8)
        # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        mask = torch.argmax(output_feature, dim=1).unsqueeze(1)
        #####################################################3unchg
        if DomainLabel==0:
            masked_featuresunchg = concat_feature_8 * (1 - mask)  # [14, 32, 256, 256])
            queryunchg = torch.sum(masked_featuresunchg, dim=(2, 3), keepdim=True) / (
                    (1 - mask).sum(dim=(2, 3), keepdim=True) + 1e-5)  # center#[14, 32, 1, 1]
            # print(queryunchg.shape)
            queryunchg=queryunchg.repeat(1,1,self.unchgN,1).reshape(output_feature.shape[0],-1,1,1)
            queryunchg = self.query_convunchg(queryunchg)  # torch.Size([14, 32, 1, 1])
            queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        else:
            # centerSunchg=Scenter[:,:,0,:].unsqueeze(-1)
            # print(Scenter.shape)
            centerSunchg=Scenter[:output_feature.shape[0],:,:self.unchgN,:].reshape(output_feature.shape[0],-1,1,1)

            masked_featuresunchg = concat_feature_8 * (1 - mask)  # [14, 32, 256, 256])
            masked_featuresunchg = torch.sum(masked_featuresunchg, dim=(2, 3), keepdim=True) / (
                    (1 - mask).sum(dim=(2, 3), keepdim=True) + 1e-5)  # center#[14, 32, 1, 1]
            # print(masked_featuresunchg.shape)
            masked_featuresunchg=masked_featuresunchg.repeat(1,1,5,1).reshape(output_feature.shape[0],-1,1,1)
            # print('centerSunchg',centerSunchg.shape,masked_featuresunchg.shape,output_feature.shape)
            centerSunchg=(centerSunchg+masked_featuresunchg)/2
            queryunchg = self.query_convunchg(centerSunchg)  # torch.Size([14, 32, 1, 1])
            queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        # Step 2: Compute similarity between the query and all locations in the feature map
        keyunchg = self.key_convunchg(concat_feature_8)  # [14, 32, 256, 256])
        keyunchg = keyunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)
        similarityunchg = torch.bmm(queryunchg, keyunchg)  # Batch matrix multiplication
        similarityunchg = similarityunchg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))
        attn_weightsunchg = F.softmax(similarityunchg, dim=1)  # [14, 14, 256, 256])

        #####################################################3chg
        if DomainLabel==0:
            masked_featureschg = concat_feature_8 * mask  # [14, 32, 256, 256])
            querychg = torch.sum(masked_featureschg, dim=(2, 3), keepdim=True) / (
                        mask.sum(dim=(2, 3), keepdim=True) + 1e-5)
            querychg=querychg.repeat(1,1,self.chgN,1).reshape(output_feature.shape[0],-1,1,1)#torch.Size([14, 160, 1, 1])
            # print('querychg',querychg.shape)

            querychg = self.query_convchg(querychg)  # torch.Size([14, 32, 1, 1])
            # print('querychg',querychg.shape)

            querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        else:
            # centerSchg=Scenter[:,:,1,:].unsqueeze(-1)
            centerSchg=Scenter[:output_feature.shape[0],:,self.chgN:,:].reshape(output_feature.shape[0],-1,1,1)
            masked_featureschg = concat_feature_8 * mask  # [14, 32, 256, 256])
            masked_featureschg = torch.sum(masked_featureschg, dim=(2, 3), keepdim=True) / (
                    mask.sum(dim=(2, 3), keepdim=True) + 1e-5)
            masked_featureschg=masked_featureschg.repeat(1,1,self.chgN,1).reshape(output_feature.shape[0],-1,1,1)
            # print('masked_featureschg',centerSchg.shape,masked_featureschg.shape)

            centerSchg=(centerSchg+masked_featureschg)/2
            querychg = self.query_convchg(centerSchg)  # torch.Size([14, 32, 1, 1])
            querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)

        keychg = self.key_convchg(concat_feature_8)  # [14, 32, 256, 256])
        keychg = keychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)
        similaritychg = torch.bmm(querychg, keychg)  # Batch matrix multiplication
        similaritychg = similaritychg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))
        # Step 3: Apply similarity weights to feature map
        attn_weightschg = F.softmax(similaritychg, dim=1)#([14, 1, 256, 256])
        # attn_features = concat_feature_8 * attn_weightsunchg+concat_feature_8 * attn_weightschg
        attn_features=concat_feature_8*(attn_weightsunchg+attn_weightschg)/2
        # print('querychg',querychg.shape,keychg.shape,concat_feature_8.shape,attn_weightschg.shape)  #querychg torch.Size([14, 1, 32]) torch.Size([14, 32, 65536]) torch.Size([14, 32, 256, 256])([14, 1, 256, 256])

        # attn_features = masked_featuresunchg * attn_weightsunchg + masked_featureschg * attn_weightschg
        output_feature = self.conv_block_8(attn_features)

        # return output_feature,{'unchgW':attn_weightsunchg, 'chgW':attn_weightschg,'cva':0,'img1L':pre_data[:, 3, :, :],'img2L':post_data[:, 3, :, :]},attn_features
        return output_feature,{'unchgW':attn_weightsunchg, 'chgW':attn_weightschg,'outF':out},attn_features

class FCSiamDiffMaskAttentionCenter2(nn.Module):
    def __init__(self, in_dim=3,out_dim=2):
        super(FCSiamDiffMaskAttentionCenter2, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )

        self.query_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.key_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.query_convchg = nn.Conv2d(32, 32, kernel_size=1)
        self.key_convchg = nn.Conv2d(32, 32, kernel_size=1)

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_1(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_1(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_1(feature_4)
        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def cva(self,T1,T2):
        difference = T2[:, 0:3, :, :] - T1[:, 0:3, :, :]
        magnitude = torch.norm(difference, dim=1)
        # direction = torch.atan2(difference[1], difference[0])
        change_map = (magnitude > 0.8).float()
        # print('change_map',change_map.shape)
        return change_map

    def forward(self, pre_data, post_data,DomainLabel=0,Scenter=None):
        #####################
        # decoder
        #####################
        # cdpre=self.cva(pre_data,post_data)
        # pre_data = pre_data[:, 0:3, :, :]
        # post_data = post_data[:, 0:3, :, :]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data[:, 0:3, :, :])
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data[:, 0:3, :, :])
        out=torch.cat([down_feature_41,down_feature_42],1)
        # out=torch.abs(down_feature_41 - down_feature_42)

        up_feature_5 = self.up_sample_1(out)
        # print('up_feature_5',up_feature_5.shape,torch.abs(feature_41 - feature_42).shape)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        DA = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([DA, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        mask = torch.argmax(output_feature, dim=1).unsqueeze(1)
        ###########3unchg
        if DomainLabel==0:
            centerSunchg=Scenter[:,:,0,:].unsqueeze(-1)

            masked_featuresunchg = concat_feature_8 * (1 - mask)  # [14, 32, 256, 256])
            queryunchg = torch.sum(masked_featuresunchg, dim=(2, 3), keepdim=True) / (
                    (1 - mask).sum(dim=(2, 3), keepdim=True) + 1e-5)  # center#[14, 32, 1, 1]
            # print(queryunchg.shape)
            queryunchg = self.query_conv(queryunchg)  # torch.Size([14, 32, 1, 1])
            queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        else:
            centerSunchg=Scenter[:,:,0,:].unsqueeze(-1)
            queryunchg = self.query_conv(centerSunchg)  # torch.Size([14, 32, 1, 1])
            queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        # Step 2: Compute similarity between the query and all locations in the feature map
        keyunchg = self.key_conv(concat_feature_8)  # [14, 32, 256, 256])
        keyunchg = keyunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)
        similarityunchg = torch.bmm(queryunchg, keyunchg)  # Batch matrix multiplication
        similarityunchg = similarityunchg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))
        # Step 3: Apply similarity weights to feature map
        attn_weightsunchg = F.softmax(similarityunchg, dim=1)  # [14, 14, 256, 256])
        ###########3chg
        if DomainLabel==0:
            centerSchg=Scenter[:,:,1,:].unsqueeze(-1)

            masked_featureschg = concat_feature_8 * mask  # [14, 32, 256, 256])
            querychg = torch.sum(masked_featureschg, dim=(2, 3), keepdim=True) / (
                        mask.sum(dim=(2, 3), keepdim=True) + 1e-5)
            querychg = self.query_convchg(querychg)  # torch.Size([14, 32, 1, 1])
            querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        else:
            centerSchg=Scenter[:,:,1,:].unsqueeze(-1)
            querychg = self.query_conv(centerSchg)  # torch.Size([14, 32, 1, 1])
            querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        # Step 2: Compute similarity between the query and all locations in the feature map
        keychg = self.key_convchg(concat_feature_8)  # [14, 32, 256, 256])
        keychg = keychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)
        similaritychg = torch.bmm(querychg, keychg)  # Batch matrix multiplication
        similaritychg = similaritychg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))
        # Step 3: Apply similarity weights to feature map
        attn_weightschg = F.softmax(similaritychg, dim=1)#[14, 14, 256, 256])
        # attn_features = concat_feature_8 * attn_weightsunchg+concat_feature_8 * attn_weightschg
        # attn_features=concat_feature_8*(attn_weightsunchg+attn_weightschg)/2

        attn_features = centerSunchg * attn_weightsunchg + centerSchg * attn_weightschg
        output_feature = self.conv_block_8(attn_features)

        # return output_feature,{'unchgW':attn_weightsunchg, 'chgW':attn_weightschg,'cva':0,'img1L':pre_data[:, 3, :, :],'img2L':post_data[:, 3, :, :]},attn_features
        return output_feature,{'unchgW':attn_weightsunchg, 'chgW':attn_weightschg,'outF':out},attn_features
class FCSiamDiffMaskAttentionCenterMask(nn.Module):
    def __init__(self, in_dim=3,out_dim=2,device=None,kernelsize=17,B=16):
        super(FCSiamDiffMaskAttentionCenterMask, self).__init__()
        self.AllDataMaskGet=AllDataMask2(orisize=256,device=device,kernelsize=kernelsize,B=B)#genMaskPatch genpatch
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.query_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.key_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.query_convchg = nn.Conv2d(32, 32, kernel_size=1)
        self.key_convchg = nn.Conv2d(32, 32, kernel_size=1)
    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_1(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_1(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_1(feature_4)
        return down_feature_4, feature_4, feature_3, feature_2, feature_1
    def forward(self, pre_data, post_data,DomainLabel=0,Scenter=None,targetimg=None):
        #####################
        # decoder
        #####################
        # cdpre=self.cva(pre_data[:, 3, :, :],post_data[:, 3, :, :])
        pre_data = pre_data[:, 0:3, :, :]
        post_data = post_data[:, 0:3, :, :]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)
        out=torch.cat([down_feature_41,down_feature_42],1)
        # out=torch.abs(down_feature_41 - down_feature_42)

        up_feature_5 = self.up_sample_1(out)
        # print('up_feature_5',up_feature_5.shape,torch.abs(feature_41 - feature_42).shape)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        DA = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([DA, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        ###########3unchg
        if DomainLabel==0:
            mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            masked_featuresunchg = concat_feature_8 * (1 - mask)  # [14, 32, 256, 256])
            queryunchg = torch.sum(masked_featuresunchg, dim=(2, 3), keepdim=True) / (
                    (1 - mask).sum(dim=(2, 3), keepdim=True) + 1e-5)  # center#[14, 32, 1, 1]
            queryunchg = self.query_conv(queryunchg)  # torch.Size([14, 32, 1, 1])
            queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        else:
            Tout = torch.argmax(output_feature, dim=1)
            targetimg['pseduoL']=Tout.detach().unsqueeze(1)
            self.AllDataMaskGet.drawBatch=5
            mask=self.AllDataMaskGet(targetimg=targetimg,infeat=output_feature.detach())##############################select sample
            centerSunchg = concat_feature_8 * (1 - (mask+0.001).detach())  # [14, 32, 256, 256])
            # centerSunchg=Scenter[:,:,0,:].unsqueeze(-1)
            queryunchg = self.query_conv(centerSunchg)  # torch.Size([14, 32, 1, 1])
            queryunchg = queryunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        # Step 2: Compute similarity between the query and all locations in the feature map
        keyunchg = self.key_conv(concat_feature_8)  # [14, 32, 256, 256])
        keyunchg = keyunchg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)
        similarityunchg = torch.bmm(queryunchg, keyunchg)  # Batch matrix multiplication
        similarityunchg = similarityunchg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))
        # Step 3: Apply similarity weights to feature map
        attn_weightsunchg = F.softmax(similarityunchg, dim=1)  # [14, 14, 256, 256])
        ###########3chg
        if DomainLabel==0:
            masked_featureschg = concat_feature_8 * mask  # [14, 32, 256, 256])
            querychg = torch.sum(masked_featureschg, dim=(2, 3), keepdim=True) / (
                        mask.sum(dim=(2, 3), keepdim=True) + 1e-5)
            querychg = self.query_convchg(querychg)  # torch.Size([14, 32, 1, 1])
            querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        else:
            centerSchg = concat_feature_8 * (mask+0.001)  # [14, 32, 256, 256])
            # centerSchg=Scenter[:,:,1,:].unsqueeze(-1)
            querychg = self.query_conv(centerSchg)  # torch.Size([14, 32, 1, 1])
            querychg = querychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1).permute(0, 2, 1)
        # Step 2: Compute similarity between the query and all locations in the feature map
        keychg = self.key_convchg(concat_feature_8)  # [14, 32, 256, 256])
        keychg = keychg.view(concat_feature_8.size(0), concat_feature_8.size(1), -1)
        similaritychg = torch.bmm(querychg, keychg)  # Batch matrix multiplication
        similaritychg = similaritychg.view(concat_feature_8.size(0), 1, concat_feature_8.size(2),
                                               concat_feature_8.size(3))
        # Step 3: Apply similarity weights to feature map
        attn_weightschg = F.softmax(similaritychg, dim=1)#[14, 14, 256, 256])
        # attn_features = concat_feature_8 * attn_weightsunchg+concat_feature_8 * attn_weightschg
        attn_features=concat_feature_8*(attn_weightsunchg+attn_weightschg)/2
        output_feature = self.conv_block_8(attn_features)

        return output_feature,[attn_weightsunchg, attn_weightschg],attn_features

class centerDist(nn.Module):
    def __init__(self,device):

        super(centerDist, self).__init__()
        dist_type = 'cos'
        self.Dist = DIST(dist_type)
        self.device=device
    def assign_labels(self, feats,filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        # print('feats', feats.sum())
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)

        if filter:
            threshold=0.3
            min_dist = torch.min(dists, dim=1)[0]  ##计算target每一类别的质心与source质心的最小距离
            mask=(min_dist<threshold).to(self.device)
            feats = torch.cat([feats[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            dists = torch.cat([dists[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            labels = torch.masked_select(labels, mask)
        return dists, labels,feats
    def forward(self,centerS,FeatureT,labelTori):
        self.centers=centerS
        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)#torch.Size([16, 16, 1024])
        labelTori=labelTori.reshape(labelTori.shape[0], -1)#([16,  1024])
        # print('labelTori',labelTori.shape,FeatureT.shape)
        # dist = F.cosine_similarity(centerS,FeatT)
        dist2CenterC=[]
        for b in range(FeatureT.shape[0]):
            FeatureTb = FeatureT[b, :, :].transpose(1, 0)
            dist2center, labels, _ = self.assign_labels(FeatureTb, filter=False)# torch.Size([1024, 2]) torch.Size([1024]) 32*32=1024
            dist2CenterC.append(dist2center.unsqueeze(0))
            # print('dist2center',dist2center.shape, labels.shape)
        dist2CenterC=torch.cat(dist2CenterC,dim=0)#[40, 1024, 2][k*b*2,kernel*kernal,2]
        # print('dist2CenterC',dist2CenterC.shape,labelTori.shape)
        unchgNum = ((1 - labelTori).sum() + 1).detach()
        chgNum = ((labelTori).sum() + 1).detach()
        chgLoss = torch.sum(dist2CenterC[:,:,1]*labelTori)/chgNum
        unchgLoss = torch.sum(dist2CenterC[:, :, 0] * (1-labelTori))/unchgNum
        LossCenter=chgLoss+unchgLoss
        # print('LossCenter',LossCenter)
        return LossCenter



class genMaskPatchNew(nn.Module):
    def __init__(self,orisize=256,device=None,kernel=32):
        super(genMaskPatchNew, self).__init__()
        self.zero = torch.tensor(0).to(device)
        self.max = torch.tensor(orisize - 1).to(device)
        self.stride = 2
        self.padding = 0
        self.kernel = kernel
        poolSize = (orisize - self.kernel) // self.stride + 1
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(poolSize).cuda(), torch.arange(poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])


        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool=nn.AvgPool2d(kernel_size=(self.kernel,self.kernel),stride=2,padding=self.padding)
    def forward(self, infeat, labelTpesudo,labelT,FeatureDA, device=None,k=1):
        output = F.softmax(infeat, dim=1)
        output_featurePatch = self.avgpool(output)
        mask=torch.zeros_like(labelT).to(device)
        # maskone=torch.ones_like(labelT).to(device)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict= {'unchg':[],'chg':[]}
        # k=3#select the first three maximum value
        for c in range(2):
            output_featurePatchum = torch.flatten(output_featurePatch[:, c, :, :].unsqueeze(1), start_dim=2,
                                                       end_dim=3)  # torch.Size([2, 1, 12769])
            values, indices = torch.topk(output_featurePatchum, k=k, dim=2, largest=True,
                                                   sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])
            for i in range(output.shape[0]):

                for j in range(k):
                    pxy=self.grid_xy[:,:,indices[i,:,j]].permute((1,2,0))# torch.Size([1, 1, 2])
                    px=pxy[0,0,0]
                    py=pxy[0,0,1]
                    ox=torch.tensor([torch.maximum(self.zero,px*self.stride-self.padding),
                                     torch.minimum(self.max,px*self.stride+32-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    oy=torch.tensor([torch.maximum(self.zero,py*self.stride-self.padding),
                                     torch.minimum(self.max,py*self.stride+32-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    mask[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1]=1
                    FeaturepatchOri=infeat[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)

                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                     # pcxy.append(oxy)
                    elementDict = {'pointXY':oxy,'c':c,'i':j}
                    if FeaturepatchOri.shape[2]==self.kernel and FeaturepatchOri.shape[3]==self.kernel and j<k:
                        if c==0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                        break
                    else:
                       print('genMaskPatchError!!')
                       continue

        return PatchDict,mask.detach()

class genMaskPatch(nn.Module):
    def __init__(self,orisize=256,device=None,kernel=32):
        super(genMaskPatch, self).__init__()
        self.zero = torch.tensor(0).to(device)
        self.max = torch.tensor(orisize - 1).to(device)
        self.stride = 2
        self.padding = 0
        self.kernel = kernel
        poolSize = (orisize - self.kernel) // self.stride + 1
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(poolSize).cuda(), torch.arange(poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool=nn.AvgPool2d(kernel_size=(self.kernel,self.kernel),stride=2,padding=self.padding)
    def forward(self, infeat, labelTpesudo,labelT,FeatureDA, device=None,k=1):
        output = F.softmax(infeat, dim=1)
        output_featurePatch = self.avgpool(output)
        mask=torch.zeros_like(labelT).to(device)
        # maskone=torch.ones_like(labelT).to(device)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict= {'unchg':[],'chg':[]}
        # k=3#select the first three maximum value
        for c in range(2):
            output_featurePatchum = torch.flatten(output_featurePatch[:, c, :, :].unsqueeze(1), start_dim=2,
                                                       end_dim=3)  # torch.Size([2, 1, 12769])
            values, indices = torch.topk(output_featurePatchum, k=k, dim=2, largest=True,
                                                   sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])
            for i in range(output.shape[0]):

                for j in range(k):
                    pxy=self.grid_xy[:,:,indices[i,:,j]].permute((1,2,0))# torch.Size([1, 1, 2])
                    px=pxy[0,0,0]
                    py=pxy[0,0,1]
                    ox=torch.tensor([torch.maximum(self.zero,px*self.stride-self.padding),
                                     torch.minimum(self.max,px*self.stride+self.kernel-1-self.padding)],#self.kernel=32
                                    dtype=torch.int32).to(device)
                    oy=torch.tensor([torch.maximum(self.zero,py*self.stride-self.padding),
                                     torch.minimum(self.max,py*self.stride+self.kernel-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    mask[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1]=1
                    FeaturepatchOri=infeat[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    patchFeat=FeatureDA[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    labelTTrue=labelT[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    labelTpesudopatch=labelTpesudo[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                     # pcxy.append(oxy)
                    elementDict = {'ClassiferT':FeaturepatchOri,'patchFeatDA':patchFeat,'provalue':values[i,:,j],'pointXY':oxy,'c':c,'i':j,
                                   'labelpesudo':labelTpesudopatch,'labelTTrue':labelTTrue}
                    if FeaturepatchOri.shape[2]==self.kernel and FeaturepatchOri.shape[3]==self.kernel and j<k:
                        if c==0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                        break
                    else:
                       print('genMaskPatchError!!')
                       continue

        return PatchDict,mask.detach()
class genpatchwithMaskEntropy(nn.Module):
    def __init__(self,orisize=256,device=None,kernel=32):
        super(genpatchwithMaskEntropy, self).__init__()
        self.kernel = kernel
        self.device = device
        self.zero = torch.tensor(0).to(self.device)
        self.stride=1
        self.padding = 0

        self.poolSize=(orisize-self.kernel)//self.stride+1
        self.max = torch.tensor(orisize - 1).to(self.device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(self.poolSize).cuda(), torch.arange(self.poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool=nn.AvgPool2d(kernel_size=(self.kernel,self.kernel),stride=self.stride,padding=self.padding)
        self.poolSize=torch.tensor(self.poolSize,dtype=torch.int32)
    def forward(self, infeat, labelTpesudo,labelT,FeatureDA, device=None,k=2):
        # output = F.softmax(infeat[:,:,self.kernel:-self.kernel,self.kernel:-self.kernel], dim=1)
        # print('labelTpesudo',labelTpesudo.shape)
        output=F.softmax(infeat)
        outputENtropy=self.Entropy(output)
        output_entropyPatch = torch.sum(self.avgpool(outputENtropy),dim=1)
        # print(output_entropyPatch.shape,output.shape)#[20, 2, 256, 256]
        output_featurePatch = self.avgpool(output)
        # labelTpesudoOneHot=F.one_hot(labelTpesudo.squeeze(1),num_classes=2)
        selectmask = torch.ones_like(output_featurePatch)
        # print('out',output_featurePatch.shape,selectmask.shape)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict= {'unchg':[],'chg':[]}
        # print('output_entropyPatchOri',output_entropyPatchOri.shape)
        # k=1
        for c in range(2):
            # output_entropyPatch=output_entropyPatchOri*labelTpesudoOneHot[:,:,:,c]
            for kk in range(k):
                # print(selectmask[:,c,self.kernel//2:-self.kernel//2+1,self.kernel//2:-self.kernel//2+1].shape,output_featurePatch[:,c,:,:].shape)
                output_featurePatchFilter=(output_featurePatch[:,c,:,:]-0.1*output_entropyPatch)*selectmask[:,c,:,:]
                # output_featurePatchFilter=(output_featurePatch[:,c,:,:])*selectmask[:,c,:,:]
                output_featurePatchum = torch.flatten(output_featurePatchFilter.unsqueeze(1), start_dim=2,
                                                           end_dim=3)  # torch.Size([2, 1, 12769])
                values, indices = torch.topk(output_featurePatchum, k=1, dim=2, largest=True,
                                                       sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])#select the first three maximum value
                for i in range(output.shape[0]):
                    j=0
                    pxy=self.grid_xy[:,:,indices[i,:,j]].permute((1,2,0))# torch.Size([1, 1, 2])
                    px=pxy[0,0,0]
                    py=pxy[0,0,1]

                    oxp = torch.tensor([torch.maximum(self.zero, px - self.kernel//self.stride//2),
                                       torch.minimum(self.poolSize, px + self.kernel//self.stride//2)],
                                      dtype=torch.int32).to(device)
                    oyp = torch.tensor([torch.maximum(self.zero, py - self.kernel // self.stride // 2),
                                       torch.minimum(self.poolSize, py + self.kernel // self.stride // 2)],
                                      dtype=torch.int32).to(device)
                    selectmask[i, c, oyp[0]:oyp[1], oxp[0]:oxp[1]] = 0
                    ox=torch.tensor([torch.maximum(self.zero,px*self.stride-self.padding),
                                     torch.minimum(self.max,px*self.stride+self.kernel-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    oy=torch.tensor([torch.maximum(self.zero,py*self.stride-self.padding),
                                     torch.minimum(self.max,py*self.stride+self.kernel-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    # outputOut=output[i,c,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    # print('in',torch.mean(outputOut),px,py,c,output_featurePatch[i,c,pyy,pxx]) #Veri

                    FeaturepatchOri=infeat[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    patchFeat=FeatureDA[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    labelTTrue=labelT[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    labelTpesudopatch=labelTpesudo[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                    elementDict = {'ClassiferT':FeaturepatchOri,'patchFeatDA':patchFeat,'provalue':values[i,:,j],'pointXY':oxy,'c':c,'i':j,
                                   'labelpesudo':labelTpesudopatch,'labelTTrue':labelTTrue}
                    if FeaturepatchOri.shape[2]==self.kernel and FeaturepatchOri.shape[3]==self.kernel:
                        if c==0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                    else:
                        print(FeaturepatchOri.shape)
                        print('genpatchError!!')
                        continue
                    # pxx = torch.tensor(pxy[0, 0, 0], dtype=torch.int32).to(device)
                    # pyy = torch.tensor(pxy[0, 0, 1], dtype=torch.int32).to(device)

                    # if output_featurePatch[i,c,pyy,pxx]<0.5:
                    #     print('values',c,kk,output_featurePatch[i,c,pyy,pxx])
        return PatchDict

    def Patch_select_gen(self,PatchDict,device):
        feat = []
        labelT = []
        labelpesudo=[]
        classiferT=[]
        for dict in PatchDict['unchg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        for dict in PatchDict['chg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        feat = torch.cat(feat, dim=0).to(device)
        labelT = torch.cat(labelT, dim=0).to(device)
        labelpesudo=torch.cat(labelpesudo, dim=0).to(device)
        classiferT=torch.cat(classiferT, dim=0).to(device)

        return feat,labelT,labelpesudo,classiferT
    def Entropy(self,input_):
        epsilon=1e-5
        # unchgP=input[:,0]
        entropy = -input_ * torch.log(input_ + epsilon)
        # entropy = torch.sum(entropy, dim=1)
        return entropy

class genpatchwithMask(nn.Module):
    def __init__(self,orisize=256,device=None,kernel=32):
        super(genpatchwithMask, self).__init__()
        self.kernel = kernel
        self.device = device
        self.zero = torch.tensor(0).to(self.device)
        self.stride = 1
        self.padding = 0

        self.poolSize = (orisize - self.kernel) // self.stride + 1
        self.max = torch.tensor(orisize - 1).to(self.device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(self.poolSize).cuda(), torch.arange(self.poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool = nn.AvgPool2d(kernel_size=(self.kernel, self.kernel), stride=self.stride, padding=self.padding)
        self.poolSize = torch.tensor(self.poolSize, dtype=torch.int32)

    def forward(self, infeat, labelTpesudo, labelT, FeatureDA, device=None, k=2):
        # output = F.softmax(infeat[:,:,self.kernel:-self.kernel,self.kernel:-self.kernel], dim=1)
        # print('labelTpesudo',labelTpesudo.shape)
        output = F.softmax(infeat)
        # print(output_entropyPatch.shape,output.shape)#[20, 2, 256, 256]
        output_featurePatch = self.avgpool(output)
        # labelTpesudoOneHot=F.one_hot(labelTpesudo.squeeze(1),num_classes=2)
        selectmask = torch.ones_like(output_featurePatch)
        # print('out',output_featurePatch.shape,selectmask.shape)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict = {'unchg': [], 'chg': []}
        # print('output_entropyPatchOri',output_entropyPatchOri.shape)
        # k=1
        for c in range(2):
            # output_entropyPatch=output_entropyPatchOri*labelTpesudoOneHot[:,:,:,c]
            for kk in range(k):
                # print(selectmask[:,c,self.kernel//2:-self.kernel//2+1,self.kernel//2:-self.kernel//2+1].shape,output_featurePatch[:,c,:,:].shape)
                output_featurePatchFilter = (output_featurePatch[:, c, :, :]) * (selectmask[:, c, :, :]).detach()
                # output_featurePatchFilter=(output_featurePatch[:,c,:,:])*selectmask[:,c,:,:]
                output_featurePatchum = torch.flatten(output_featurePatchFilter.unsqueeze(1), start_dim=2,end_dim=3)  # torch.Size([2, 1, 12769])
                # select the first three maximum value
                values, indices = torch.topk(output_featurePatchum, k=1, dim=2, largest=True,
                                             sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])
                for i in range(output.shape[0]):
                    j = 0
                    pxy = self.grid_xy[:, :, indices[i, :, j]].permute((1, 2, 0))  # torch.Size([1, 1, 2])
                    px = pxy[0, 0, 0]
                    py = pxy[0, 0, 1]

                    oxp = torch.tensor([torch.maximum(self.zero, px - self.kernel // self.stride // 2),
                                        torch.minimum(self.poolSize, px + self.kernel // self.stride // 2)],
                                       dtype=torch.int32).to(device)
                    oyp = torch.tensor([torch.maximum(self.zero, py - self.kernel // self.stride // 2),
                                        torch.minimum(self.poolSize, py + self.kernel // self.stride // 2)],
                                       dtype=torch.int32).to(device)
                    selectmask[i, c, oyp[0]:oyp[1], oxp[0]:oxp[1]] = 0
                    ox = torch.tensor([torch.maximum(self.zero, px * self.stride - self.padding),
                                       torch.minimum(self.max, px * self.stride + self.kernel - 1 - self.padding)],
                                      dtype=torch.int32).to(device)
                    oy = torch.tensor([torch.maximum(self.zero, py * self.stride - self.padding),
                                       torch.minimum(self.max, py * self.stride + self.kernel - 1 - self.padding)],
                                      dtype=torch.int32).to(device)
                    # outputOut=output[i,c,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    # print('in',torch.mean(outputOut),px,py,c,output_featurePatch[i,c,pyy,pxx]) #Veri

                    FeaturepatchOri = infeat[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    patchFeat = FeatureDA[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    labelTTrue = labelT[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    labelTpesudopatch = labelTpesudo[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                    elementDict = {'ClassiferT': FeaturepatchOri, 'patchFeatDA': patchFeat, 'provalue': values[i, :, j],
                                   'pointXY': oxy, 'c': c, 'i': j,
                                   'labelpesudo': labelTpesudopatch, 'labelTTrue': labelTTrue}
                    if FeaturepatchOri.shape[2] == self.kernel and FeaturepatchOri.shape[3] == self.kernel:
                        if c == 0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                    else:
                        print(FeaturepatchOri.shape)
                        print('genpatchError!!')
                        continue
                    # pxx = torch.tensor(pxy[0, 0, 0], dtype=torch.int32).to(device)
                    # pyy = torch.tensor(pxy[0, 0, 1], dtype=torch.int32).to(device)

                    # if output_featurePatch[i,c,pyy,pxx]<0.5:
                    #     print('values',c,kk,output_featurePatch[i,c,pyy,pxx])
        return PatchDict
    def Patch_select_gen(self,PatchDict,device):
        feat = []
        labelT = []
        labelpesudo=[]
        classiferT=[]
        for dict in PatchDict['unchg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        for dict in PatchDict['chg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        feat = torch.cat(feat, dim=0).to(device)
        labelT = torch.cat(labelT, dim=0).to(device)
        labelpesudo=torch.cat(labelpesudo, dim=0).to(device)
        classiferT=torch.cat(classiferT, dim=0).to(device)

        return feat,labelT,labelpesudo,classiferT
class AllDataMask(nn.Module):
    def __init__(self,orisize=256,device=None,kernelsize=32,B=28):
        super(AllDataMask, self).__init__()
        self.kernelsize = kernelsize
        self.device = device
        self.zero = torch.tensor(0).to(self.device)
        self.stride = 1
        self.padding = 0
        self.poolSize = (orisize - self.kernelsize) // self.stride + 1
        self.max = torch.tensor(orisize - 1).to(self.device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(self.poolSize).cuda(), torch.arange(self.poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)#torch.Size([2, 1, 58081])
        self.avgpool = nn.AvgPool2d(kernel_size=(self.kernelsize, self.kernelsize), stride=self.stride, padding=self.padding)
        self.poolSize = torch.tensor(self.poolSize, dtype=torch.int32)

        self.expansionKernal = torch.zeros((B, 1, self.kernelsize * 2 + 1, self.kernelsize * 2 + 1)).to(device)
        self.expansionKernal[:, :, 0:self.kernelsize + 1, 0:self.kernelsize + 1] = 1
        self.name={'0': 'Unchg', '1': 'Chg'}
        # self.expansionKernal=torch.tensor(expansionKernal,dtype=torch.float32)
        self.ones = torch.ones((B, self.poolSize, self.poolSize)).to(device)
        self.zeros = torch.zeros((B, self.poolSize, self.poolSize)).to(device)
    def plot(self,targetimg,output_maskOut,softmaxOut,maskbefore,c=0):
        f=15
        if c==0:
            img1= np.array((targetimg['img1'][f,0:3,:,:]).permute(1,2,0).detach().cpu().numpy()* 255, dtype=np.uint8)
            img2=np.array((targetimg['img2'][f,0:3,:,:]).permute(1,2,0).detach().cpu().numpy()* 255, dtype=np.uint8)
            labelT=np.array((targetimg['labelT'][f,:,:]).permute(1,2,0).detach().cpu().numpy()* 255, dtype=np.uint8)
            pseduoL=np.array((targetimg['pseduoL'][f,:,:]).permute(1,2,0).detach().cpu().numpy()* 255, dtype=np.uint8)
            # cv2.imwrite('./log/out/%d_1img1.png' % targetimg['i'], img1)
            # cv2.imwrite('./log/out/%d_1img2.png' % targetimg['i'], img2)
            # cv2.imwrite('./log/out/%d_1label.png' % targetimg['i'], labelT)
            # cv2.imwrite('./log/out/%d_1pseduoL.png' % targetimg['i'], pseduoL)
            cv2.imwrite('./log/out/img/%d_1img.png' % targetimg['i'], np.concatenate([img1,img2],axis=1))
            cv2.imwrite('./log/out/img/%d_1label.png' % targetimg['i'], np.concatenate([labelT,pseduoL],axis=1))
            self.saveplot['img1']=targetimg['img1'].detach().cpu().numpy()
            self.saveplot['img2'] = targetimg['img2'].detach().cpu().numpy()
            self.saveplot['labelT'] = targetimg['labelT'].detach().cpu().numpy()
            self.saveplot['softmaxOut']=softmaxOut.detach().cpu().numpy()
            np.save('./log/out/dict/%d.npy'%targetimg['i'], dict)  # 注意带上后缀名

        softmaxOut = softmaxOut[f, c, :, :].detach().cpu().numpy()
        heatmap = cv2.applyColorMap(np.uint8(softmaxOut * 255), cv2.COLORMAP_JET)
        maskbefore=maskbefore[f,:,:].detach().cpu().numpy()
        output_maskOut = output_maskOut[f, 0, :, :].detach().cpu().numpy()
        cm = plt.cm.get_cmap('jet')
        plt.imshow(softmaxOut, cmap=cm)
        plt.colorbar()
        plt.savefig('./log/out/img/%d_2heatmap2%s.png'%(targetimg['i'],self.name[str(c)]))
        plt.clf()
        cv2.imwrite('./log/out/img/%d_2heatmap%s.png'%(targetimg['i'],self.name[str(c)]),np.array(heatmap,dtype=np.uint8))
        cv2.imwrite('./log/out/img/%d_3mask%sBefore.png'%(targetimg['i'],self.name[str(c)]),np.array(maskbefore*255,dtype=np.uint8))
        cv2.imwrite('./log/out/img/%d_4mask%s.png'%(targetimg['i'],self.name[str(c)]),np.array(output_maskOut*255,dtype=np.uint8))
        # cv2.imwrite('./log/out/%d_3out.png' % targetimg['i'], np.concatenate([np.array(maskbefore*255,dtype=np.uint8),np.array(output_maskOut*255,dtype=np.uint8)], axis=1))

    def forward(self, targetimg,infeat, FeatureDA, device=None, k=2):
        output = F.softmax(infeat,dim=1)
        b,c,w,h=infeat.shape
        output_featurePatch = self.avgpool(output)
        # selectmask = torch.ones_like(output_featurePatch)
        PatchDict = {'unchg': [], 'chg': []}
        self.saveplot={}
        print('output_featurePatch.shape',output_featurePatch.shape,self.poolSize)
        for c in range(2):
            # ones=torch.ones((output_featurePatch.shape[0],output_featurePatch.shape[2],output_featurePatch.shape[3])).to(device)
            # zeros=torch.zeros((output_featurePatch.shape[0],output_featurePatch.shape[2],output_featurePatch.shape[3])).to(device)
            output_featurePatchFilterMask=torch.where(output_featurePatch[:, c, :, :]>0.5,self.ones,self.zeros)#torch.Size([28, 241, 241])
            output_featurePatchFilterMaskP = F.pad(output_featurePatchFilterMask, (self.kernelsize, self.kernelsize * 2 - 1, self.kernelsize, self.kernelsize * 2 - 1),
                                                  'constant', 0)
            zerosMask=torch.zeros(b,w,h).to(device)
            output_mask = F.conv2d(output_featurePatchFilterMaskP.unsqueeze(1), self.expansionKernal, padding=0)
            output_maskbool = output_mask.bool()
            output_maskOut = zerosMask.masked_fill(output_maskbool, 1)
            print('output_maskOut',output_maskOut.shape)
            self.plot(targetimg=targetimg,output_maskOut=output_maskOut,softmaxOut=output,maskbefore=output_featurePatchFilterMask,c=c)

        return PatchDict

class AllDataMask2(nn.Module):
    def __init__(self,orisize=256,device=None,kernelsize=32,B=28):
        super(AllDataMask2, self).__init__()
        self.kernelsize = kernelsize
        self.device = device
        self.marginMask = torch.ones((B, 1, orisize, orisize), device=device)
        self.marginMask[:, :, 0:kernelsize // 2, :] = 0
        self.marginMask[:, :, -kernelsize // 2:, :] = 0
        self.marginMask[:, :, :, 0:kernelsize // 2] = 0
        self.marginMask[:, :, :, -kernelsize // 2:] = 0

        MeanKernal = torch.ones(1, 1, kernelsize, kernelsize, device=device) / (kernelsize * kernelsize)
        self.MeanKernal = nn.Parameter(MeanKernal, requires_grad=False)

        expandsionKernal = torch.ones(1, 1, kernelsize, kernelsize, device=device)
        self.expandsionKernal = nn.Parameter(expandsionKernal, requires_grad=False)


        self.zeroArray = torch.zeros((B,orisize,orisize)).to(self.device)
        self.oneArray = torch.ones((B,orisize,orisize)).to(self.device)

        self.name = {'0': 'Unchg', '1': 'Chg'}


    def plot(self,targetimg,output_maskOut,softmaxOut,maskbefore):
        if not os.path.exists('./log/out/img'):
            os.makedirs('./log/out/img')
        self.drawBatch=1
        # f=self.f
        img1= np.array((targetimg['img1'][self.drawBatch,0:3,:,:]).permute(1,2,0).detach().cpu().numpy()* 255, dtype=np.uint8)
        img2=np.array((targetimg['img2'][self.drawBatch,0:3,:,:]).permute(1,2,0).detach().cpu().numpy()* 255, dtype=np.uint8)
        # print('targetimg',targetimg['img1L'].shape)
        # img1L = np.array((targetimg['img1L'][self.drawBatch, :, :]).detach().cpu().numpy() * 255,
        #                 dtype=np.uint8)
        # img2L = np.array((targetimg['img2L'][self.drawBatch, :, :]).detach().cpu().numpy() * 255,
        #                 dtype=np.uint8)
        labelT=np.array((targetimg['labelT'][self.drawBatch,0,:,:]).detach().cpu().numpy()* 255, dtype=np.uint8)
        pseduoL=np.array((targetimg['pseduoL'][self.drawBatch,0,:,:]).detach().cpu().numpy()* 255, dtype=np.uint8)
        # cvaL=np.array((targetimg['cva'][self.drawBatch,:,:]).detach().cpu().numpy()* 255, dtype=np.uint8)
        cv2.imwrite('./log/out/img/%d_1img.png' % (targetimg['i']), np.concatenate([img1,img2],axis=1))
        # cv2.imwrite('./log/out/img/%d_1imgL.png' % (targetimg['i']), np.concatenate([img1L,img2L],axis=1))
        cv2.imwrite('./log/out/img/%d_1label.png' % (targetimg['i']), np.concatenate([labelT,pseduoL],axis=1))
        # self.saveplot['img1']=targetimg['img1'].detach().cpu().numpy()
        # self.saveplot['img2'] = targetimg['img2'].detach().cpu().numpy()
        # self.saveplot['labelT'] = targetimg['labelT'].detach().cpu().numpy()
        # self.saveplot['softmaxOut']=softmaxOut.detach().cpu().numpy()
        np.save('./log/out/dict/%d.npy'%(targetimg['i']), dict)  # 注意带上后缀名
        for c in range(2):
            softmaxOut_ = softmaxOut[self.drawBatch, c, :, :].detach().cpu().numpy()
            heatmap = cv2.applyColorMap(np.uint8(softmaxOut_ * 255), cv2.COLORMAP_JET)
            maskbefore_=maskbefore[str(c)][self.drawBatch,0,:,:].detach().cpu().numpy()
            output_maskOut_ = output_maskOut[str(c)][self.drawBatch, 0, :, :].detach().cpu().numpy()
            cm = plt.cm.get_cmap('jet')
            plt.imshow(softmaxOut_, cmap=cm,vmin=0,vmax=1)
            plt.colorbar()
            plt.savefig('./log/out/img/%d_2heatmap2%s.png'%(targetimg['i'],self.name[str(c)]))
            plt.clf()
            dst1 = np.array(cv2.addWeighted(img1, 0.6, heatmap, 0.4, 0),dtype=np.uint8)
            dst2 = np.array(cv2.addWeighted(img2, 0.6, heatmap, 0.4, 0),dtype=np.uint8)
            cv2.imwrite('./log/out/img/%d_2heatmap%s.png'%(targetimg['i'],self.name[str(c)]),np.concatenate([dst1,dst2],axis=1))
            cv2.imwrite('./log/out/img/%d_3mask%sBefore.png'%(targetimg['i'],self.name[str(c)]),np.array(maskbefore_*255,dtype=np.uint8))
            cv2.imwrite('./log/out/img/%d_4mask%s.png'%(targetimg['i'],self.name[str(c)]),np.array(output_maskOut_*255,dtype=np.uint8))
        # cv2.imwrite('./log/out/%d_3out.png' % targetimg['i'], np.concatenate([np.array(maskbefore*255,dtype=np.uint8),np.array(output_maskOut*255,dtype=np.uint8)], axis=1))
        intersection = (output_maskOut['0'] * output_maskOut['1']).squeeze(1).detach().cpu().numpy()
        union = torch.clamp(output_maskOut['0'] + output_maskOut['1'], 0, 1).squeeze(1).detach().cpu().numpy()
        print('intersection',intersection.shape,union.shape)
        # cv2.imwrite('./log/out/img/%d_5intersection.png' % (targetimg['i']),
        #             np.array(intersection[self.drawBatch] * 255, dtype=np.uint8))
        cv2.imwrite('./log/out/img/%d_5union.png' % (targetimg['i']),
                    np.array(union[self.drawBatch] * 255, dtype=np.uint8))
        # out=np.zeros_like(img1)

    def forward(self, targetimg,infeat):
        Pro = F.softmax(infeat, dim=1)
        outputEntropy = -Pro * torch.log(Pro + 1e-10)  # lower better
        Consistence_index = Pro
        b, c, w, h = infeat.shape
        mean_prob = nn.functional.conv2d(Consistence_index, self.MeanKernal.repeat(c, 1, 1, 1),
                                         padding=self.kernelsize // 2, groups=c)
        var_prob = nn.functional.conv2d(Consistence_index * Consistence_index, self.MeanKernal.repeat(c, 1, 1, 1),
                                        padding=self.kernelsize // 2, groups=c) - mean_prob ** 2
        # print('mean_prob',self.kernelsize,infeat.shape,mean_prob.shape)
        mask = {}
        maskbefor = {}
        for c in range(2):
            # self.saveplot = {}
            if c == 0:
                Thr = 0.9
            else:
                Thr = 0.6
            SelectPixel = torch.where(mean_prob[:, c, :, :] > Thr, self.oneArray, self.zeroArray).unsqueeze(
                1)  ##([28, 1, 256, 256])
            SelectRegion = SelectPixel * self.marginMask
            # SelectRegion=(F.conv2d(SelectPixel,self.expandsionKernal,padding=self.expandsionKernal.size(2)//2)>0).float()*self.marginMask#([28, 1, 256, 256])
            # print('SelectRegion',SelectRegion.shape,SelectPixel.shape)
            SelectMaskOut = SelectRegion
            mask[str(c)] = SelectMaskOut
            maskbefor[str(c)] = SelectPixel

        intersection = mask['0'] * mask['1']
        union = torch.clamp(mask['0'] + mask['1'], 0, 1)
        self.plot(targetimg=targetimg, output_maskOut=mask, softmaxOut=Pro, maskbefore=maskbefor)

        return union

class AllDataMask3(nn.Module):
    def __init__(self,orisize=256,device=None,kernelsize=32,B=28):
        super(AllDataMask3, self).__init__()
        self.kernelsize = kernelsize
        self.device = device
        self.marginMask = torch.ones((B, 1, orisize, orisize), device=device)
        self.marginMask[:, :, 0:kernelsize // 2, :] = 0
        self.marginMask[:, :, -kernelsize // 2:, :] = 0
        self.marginMask[:, :, :, 0:kernelsize // 2] = 0
        self.marginMask[:, :, :, -kernelsize // 2:] = 0

        MeanKernal = torch.ones(1, 1, kernelsize, kernelsize, device=device) / (kernelsize * kernelsize)
        self.MeanKernal = nn.Parameter(MeanKernal, requires_grad=False)

        expandsionKernal = torch.ones(1, 1, kernelsize, kernelsize, device=device)
        self.expandsionKernal = nn.Parameter(expandsionKernal, requires_grad=False)

        self.zeroArray = torch.zeros((B,orisize,orisize)).to(self.device)
        self.oneArray = torch.ones((B,orisize,orisize)).to(self.device)
        self.name = {'0': 'Unchg', '1': 'Chg'}
        kernel_size=5
        self.edKernal= torch.ones((1, 1, kernel_size, kernel_size), device=device)
    def plot(self,targetimg,output_maskOut,softmaxOut,maskbefore):
        cm = plt.cm.get_cmap('jet')
        self.drawBatch = 6
        img1 = targetimg['img1'][:self.drawBatch, 0:3, :, :]
        img2 = targetimg['img2'][:self.drawBatch, 0:3, :, :]
        labelT = targetimg['labelT'][:self.drawBatch, 0:3, :, :]
        pseudoT = targetimg['Pseudo'][:self.drawBatch, 0:3, :, :]
        # print('pseudoT',pseudoT.shape)
        [Pro, mean_prob, var_prob, outputEntropy, mean_entropy, var_entropy] = softmaxOut
        Pro = Pro[:self.drawBatch].detach().cpu().numpy()
        mean_prob = mean_prob[:self.drawBatch].detach().cpu().numpy()
        var_prob = var_prob[:self.drawBatch].detach().cpu().numpy()
        outputEntropy = outputEntropy[:self.drawBatch].detach().cpu().numpy()
        mean_entropy = mean_entropy[:self.drawBatch].detach().cpu().numpy()
        var_entropy = var_entropy[:self.drawBatch].detach().cpu().numpy()
        output_maskOut=output_maskOut[:self.drawBatch]
        print('output_maskOut',output_maskOut.shape)
        entropyD = targetimg['D1'][:self.drawBatch,0].detach().cpu().numpy()
        # print('entropyD',entropyD.shape,mean_entropy.shape)
        midFeatD = targetimg['D2'][:self.drawBatch,0].detach().cpu().numpy()
        title = ['Target T1', 'Target T2', 'Target Label', 'Pseudo Label','Out Label', 'Predict Probability', 'Predict Probability Mean',
                  'Predict Probability Variance', 'Predict Entropy', 'Predict Entropy Mean', 'Predict Entropy Variance',
                  'Mid-feature Discriminator','Entropy Discriminator']
        N=len(title)
        image = [img1, img2, labelT, pseudoT,output_maskOut, Pro, mean_prob, var_prob, outputEntropy, mean_entropy, var_entropy,midFeatD,entropyD]
        titles=[]
        images=[]
        for i in range(self.drawBatch * N):
            titles.append(title[(i) % N])
            images.append(image[(i) % N][(i)// N])
            # if (i + 1) % 9 == 0:

        fig, axes = plt.subplots(self.drawBatch, N, figsize=(35, 18))  # 假设每张子图为256x256像素，所以整体尺寸为10.24x7.68英寸
        dN=0
        for ax, img in zip(axes.ravel(), images):
            if titles[dN] in ['Target T1', 'Target T2', 'Target Label','Pseudo Label','Out Label']:
                ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                ax.set_title(titles[dN])

            elif titles[dN] in ['Mid-feature Discriminator','Entropy Discriminator']:
                # print('img',img.shape)
                im = ax.imshow(img, cmap=cm, vmin=0, vmax=1)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.10)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置colorbar标签
            elif titles[dN] in ['Predict Probability', 'Predict Probability Mean','Predict Entropy',
                                'Predict Entropy Mean']:
                im = ax.imshow(img[1], cmap=cm, vmin=0, vmax=1)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.10)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置colorbar标签
                # fig.colorbar(im,cax=cbar_ax)
            elif titles[dN] in ['Predict Probability Variance', 'Predict Entropy Variance']:
                im = ax.imshow(img[1], cmap=cm, vmin=0, vmax=0.2)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="4%", pad=0.05)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0,0.04,0.08,0.12,0.16,0.2]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(["0","0.04","0.08","0.12","0.16","0.20"])  # 设置colorbar标签

            ax.axis('off')  # 去除横纵坐标
            dN += 1
        plt.figure(num=1)
        plt.savefig('./log/out/SelectG-L/%d.png' % (targetimg['i']), dpi=100, bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)  # 关闭当前图形

    def erosion_dilation(self,mask, kernel_size=9, erosion_iter=1, dilation_iter=1):
        # 腐蚀操作
        self.edKernal= torch.ones((1, 1, kernel_size, kernel_size)).to(mask.device)

        for _ in range(erosion_iter):
            padding = kernel_size // 2
            mask = F.conv2d(mask, self.edKernal, padding=padding)
            mask = (mask == kernel_size ** 2).float()

        # # 膨胀操作
        # for _ in range(dilation_iter):
        #     padding = kernel_size // 2
        #     mask = F.conv2d(mask, self.edKernal, padding=padding)
        #     mask = (mask >= 1).float()
        for _ in range(dilation_iter):
            padding = (kernel_size - 1) // 2
            mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        return mask

    def erode_dilate_opencv(self,mask, kernel_size=11, erosion_iter=1, dilation_iter=1):
        # 将PyTorch张量转换为NumPy数组
        numpy_array = mask.detach().cpu().numpy()
        output = np.zeros_like(numpy_array)

        # 定义腐蚀和膨胀的核
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 对每个图像进行腐蚀和膨胀操作
        for i in range(numpy_array.shape[0]):
            for j in range(numpy_array.shape[1]):
                # 腐蚀
                erosion = cv2.erode(numpy_array[i, j], kernel, iterations=erosion_iter)
                # 膨胀
                dilation = cv2.dilate(erosion, kernel, iterations=dilation_iter)
                output[i, j] = dilation

        # 将NumPy数组转换回PyTorch张量
        tensor_output = torch.from_numpy(output).to(mask.device).float()
        return tensor_output
    def forward(self, targetimg,infeat):
        Pro = F.softmax(infeat, dim=1)
        outputEntropy = -Pro * torch.log2(Pro + 1e-10)  # lower better
        # print('outputEntropy',outputEntropy.shape)
        Consistence_index = Pro
        b, c, w, h = infeat.shape
        mean_prob = nn.functional.conv2d(Consistence_index, self.MeanKernal.repeat(c, 1, 1, 1),
                                         padding=self.kernelsize // 2, groups=c)
        var_prob = nn.functional.conv2d(Consistence_index * Consistence_index, self.MeanKernal.repeat(c, 1, 1, 1),
                                        padding=self.kernelsize // 2, groups=c) - mean_prob ** 2

        mean_entropy = nn.functional.conv2d(outputEntropy, self.MeanKernal.repeat(c, 1, 1, 1),
                                         padding=self.kernelsize // 2, groups=c)
        var_entropy = nn.functional.conv2d(outputEntropy * outputEntropy, self.MeanKernal.repeat(c, 1, 1, 1),
                                        padding=self.kernelsize // 2, groups=c) - mean_entropy ** 2
        # print('mean_prob',self.kernelsize,infeat.shape,mean_prob.shape)
        mask = {}
        maskbefor = {}
        # print('targetimg',targetimg['Pseudo'][:, 0, :, :].unsqueeze(1).shape)
        outPesudo=self.erode_dilate_opencv(mask= targetimg['Pseudo'][:, 0, :, :].float().unsqueeze(1))
        outPesudo = (outPesudo > 0.5).float()
        for c in range(2):
            # self.saveplot = {}
            if c == 0:
                Thr = 0.9
            else:
                Thr = 0.7
            SelectPixel = torch.where(mean_prob[:, c, :, :] > Thr, self.oneArray, self.zeroArray).unsqueeze(
                1)  ##([28, 1, 256, 256])
            # SelectRegion = SelectPixel * self.marginMask
            # SelectRegion = SelectPixel
            # SelectRegion=(F.conv2d(SelectPixel,self.expandsionKernal,padding=self.expandsionKernal.size(2)//2)>0).float()*self.marginMask#([28, 1, 256, 256])
            # print('SelectRegion',SelectRegion.shape,SelectPixel.shape)
            SelectMaskOut = SelectPixel
            # print('SelectRegion',SelectRegion.shape)
            mask[str(c)] = SelectMaskOut
            maskbefor[str(c)] = SelectPixel

        intersection = mask['0'] * mask['1']
        union = torch.clamp(mask['0'] + mask['1'], 0, 1)
        # self.plot(targetimg=targetimg, output_maskOut=outPesudo,
        #           softmaxOut=[Pro,mean_prob,var_prob,outputEntropy,mean_entropy,var_entropy], maskbefore=maskbefor)

        return union

class pseudoMultiCenter(nn.Module):
    def __init__(self,orisize=256,device=None,kernelsize=32,B=28):
        super(pseudoMultiCenter, self).__init__()
        self.kernelsize = kernelsize
        self.device = device
        # self.marginMask = torch.ones((B, 1, orisize, orisize), device=device)
        # self.marginMask[:, :, 0:kernelsize // 2, :] = 0
        # self.marginMask[:, :, -kernelsize // 2:, :] = 0
        # self.marginMask[:, :, :, 0:kernelsize // 2] = 0
        # self.marginMask[:, :, :, -kernelsize // 2:] = 0

        MeanKernal = torch.ones(1, 1, kernelsize, kernelsize, device=device) / (kernelsize * kernelsize)
        self.MeanKernal = nn.Parameter(MeanKernal, requires_grad=False)

        expandsionKernal = torch.ones(1, 1, kernelsize, kernelsize, device=device)
        self.expandsionKernal = nn.Parameter(expandsionKernal, requires_grad=False)

        self.zeroArray = torch.zeros((B,orisize,orisize)).to(self.device)
        self.oneArray = torch.ones((B,orisize,orisize)).to(self.device)
        self.name = {'0': 'Unchg', '1': 'Chg'}
        kernel_size=5
        self.edKernal= torch.ones((1, 1, kernel_size, kernel_size), device=device)
    def plot(self,targetimg,output_maskOut,softmaxOut,maskbefore):
        cm = plt.cm.get_cmap('jet')
        self.drawBatch = 6
        img1 = targetimg['img1'][:self.drawBatch, 0:3, :, :]
        img2 = targetimg['img2'][:self.drawBatch, 0:3, :, :]
        labelT = targetimg['labelT'][:self.drawBatch, 0:3, :, :]
        pseudoT = targetimg['Pseudo'][:self.drawBatch, 0:3, :, :]
        # print('pseudoT',pseudoT.shape)
        [Pro, mean_prob, var_prob, outputEntropy, mean_entropy, var_entropy] = softmaxOut
        Pro = Pro[:self.drawBatch].detach().cpu().numpy()
        mean_prob = mean_prob[:self.drawBatch].detach().cpu().numpy()
        var_prob = var_prob[:self.drawBatch].detach().cpu().numpy()
        outputEntropy = outputEntropy[:self.drawBatch].detach().cpu().numpy()
        mean_entropy = mean_entropy[:self.drawBatch].detach().cpu().numpy()
        var_entropy = var_entropy[:self.drawBatch].detach().cpu().numpy()
        output_maskOut=output_maskOut[:self.drawBatch]
        print('output_maskOut',output_maskOut.shape)
        entropyD = targetimg['D1'][:self.drawBatch,0].detach().cpu().numpy()
        # print('entropyD',entropyD.shape,mean_entropy.shape)
        midFeatD = targetimg['D2'][:self.drawBatch,0].detach().cpu().numpy()
        title = ['Target T1', 'Target T2', 'Target Label', 'Pseudo Label','Out Label', 'Predict Probability', 'Predict Probability Mean',
                  'Predict Probability Variance', 'Predict Entropy', 'Predict Entropy Mean', 'Predict Entropy Variance',
                  'Mid-feature Discriminator','Entropy Discriminator']
        N=len(title)
        image = [img1, img2, labelT, pseudoT,output_maskOut, Pro, mean_prob, var_prob, outputEntropy, mean_entropy, var_entropy,midFeatD,entropyD]
        titles=[]
        images=[]
        for i in range(self.drawBatch * N):
            titles.append(title[(i) % N])
            images.append(image[(i) % N][(i)// N])
            # if (i + 1) % 9 == 0:

        fig, axes = plt.subplots(self.drawBatch, N, figsize=(35, 18))  # 假设每张子图为256x256像素，所以整体尺寸为10.24x7.68英寸
        dN=0
        for ax, img in zip(axes.ravel(), images):
            if titles[dN] in ['Target T1', 'Target T2', 'Target Label','Pseudo Label','Out Label']:
                ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                ax.set_title(titles[dN])

            elif titles[dN] in ['Mid-feature Discriminator','Entropy Discriminator']:
                # print('img',img.shape)
                im = ax.imshow(img, cmap=cm, vmin=0, vmax=1)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.10)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置colorbar标签
            elif titles[dN] in ['Predict Probability', 'Predict Probability Mean','Predict Entropy',
                                'Predict Entropy Mean']:
                im = ax.imshow(img[1], cmap=cm, vmin=0, vmax=1)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.10)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置colorbar标签
                # fig.colorbar(im,cax=cbar_ax)
            elif titles[dN] in ['Predict Probability Variance', 'Predict Entropy Variance']:
                im = ax.imshow(img[1], cmap=cm, vmin=0, vmax=0.2)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="4%", pad=0.05)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0,0.04,0.08,0.12,0.16,0.2]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(["0","0.04","0.08","0.12","0.16","0.20"])  # 设置colorbar标签

            ax.axis('off')  # 去除横纵坐标
            dN += 1
        plt.figure(num=1)
        plt.savefig('./log/out/SelectG-L/%d.png' % (targetimg['i']), dpi=100, bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)  # 关闭当前图形

    def erosion_dilation(self,mask, kernel_size=9, erosion_iter=1, dilation_iter=1):
        # 腐蚀操作
        self.edKernal= torch.ones((1, 1, kernel_size, kernel_size)).to(mask.device)

        for _ in range(erosion_iter):
            padding = kernel_size // 2
            mask = F.conv2d(mask, self.edKernal, padding=padding)
            mask = (mask == kernel_size ** 2).float()

        # # 膨胀操作
        # for _ in range(dilation_iter):
        #     padding = kernel_size // 2
        #     mask = F.conv2d(mask, self.edKernal, padding=padding)
        #     mask = (mask >= 1).float()
        for _ in range(dilation_iter):
            padding = (kernel_size - 1) // 2
            mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        return mask
    def plotLabel(self,targetimg,pesudoCenter,pesudoCenter2,outputEntropy,mean_entropy,selectRegion,Pro):
        cm = plt.cm.get_cmap('jet')
        self.drawBatch = 6
        img1 = targetimg['img1'][:self.drawBatch, 0:3, :, :]
        img2 = targetimg['img2'][:self.drawBatch, 0:3, :, :]
        labelT = targetimg['labelT'][:self.drawBatch, 0:3, :, :]
        pseudoT = targetimg['Pseudo'][:self.drawBatch, 0:3, :, :]
        pesudoCenter=pesudoCenter[:self.drawBatch, 0, :, :].unsqueeze(1)
        pesudoCenter2=pesudoCenter2[:self.drawBatch, 0, :, :].unsqueeze(1)
        selectRegion=selectRegion[:self.drawBatch].unsqueeze(1)

        Probability = Pro[:self.drawBatch,1].detach().cpu().numpy()

        mean_entropy = mean_entropy[:self.drawBatch].sum(1).detach().cpu().numpy()
        # print('mean_entropy',mean_entropy.shape)
        outputEntropy = outputEntropy[:self.drawBatch].sum(1).detach().cpu().numpy()
        # print('pesudoCenter2',pesudoCenter2.shape)
        # print(selectRegion.shape,selectRegion.max())
        title = ['Target T1', 'Target T2', 'Target Label', 'Pseudo Label', 'Center Label','Center Label2','Probability','outputEntropy','mean_entropy','Select Region']
        image = [img1, img2, labelT, pseudoT, pesudoCenter,pesudoCenter2,Probability,outputEntropy,mean_entropy,selectRegion]
        titles = []
        images = []
        N=len(title)

        for i in range(self.drawBatch * N):
            titles.append(title[(i) % N])
            images.append(image[(i) % N][(i) // N])
            # if (i + 1) % 9 == 0:

        fig, axes = plt.subplots(self.drawBatch, N, figsize=(30, 18))  # 假设每张子图为256x256像素，所以整体尺寸为10.24x7.68英寸
        dN = 0
        for ax, img in zip(axes.ravel(), images):
            if titles[dN] in ['Target T1', 'Target T2', 'Target Label','Pseudo Label','Center Label','Center Label2','Select Region']:
                # print('titles[dN]',titles[dN])
                ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                ax.set_title(titles[dN])
            elif titles[dN] in ['outputEntropy','mean_entropy','Probability']:
                # print('img',img.shape)
                im = ax.imshow(img, cmap=cm, vmin=0, vmax=1)
                ax.set_title(titles[dN])
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.10)
                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置colorbar标签
            ax.axis('off')  # 去除横纵坐标
            dN += 1

        plt.figure(num=1)
        plt.savefig('./log/out/multiCenter/%d_%d.png' % (targetimg['epoch'],targetimg['i']), dpi=100, bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)  # 关闭当前图形
    def erode_dilate_opencv(self,mask, kernel_size=11, erosion_iter=1, dilation_iter=1):
        # 将PyTorch张量转换为NumPy数组
        numpy_array = mask.detach().cpu().numpy()
        output = np.zeros_like(numpy_array)

        # 定义腐蚀和膨胀的核
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 对每个图像进行腐蚀和膨胀操作
        for i in range(numpy_array.shape[0]):
            for j in range(numpy_array.shape[1]):
                # 腐蚀
                erosion = cv2.erode(numpy_array[i, j], kernel, iterations=erosion_iter)
                # 膨胀
                dilation = cv2.dilate(erosion, kernel, iterations=dilation_iter)
                output[i, j] = dilation

        # 将NumPy数组转换回PyTorch张量
        tensor_output = torch.from_numpy(output).to(mask.device).float()
        return tensor_output

    def forward(self, targetimg,infeat,multiCenter):
        Pro = F.softmax(infeat, dim=1)
        outputEntropy = -Pro * torch.log2(Pro + 1e-10)  # lower better
        b, c, w, h = infeat.shape
        multiCenter=multiCenter[0,:,:,0]
        # targetCenter=targetimg['defeatTDA'].view(b,targetimg['defeatTDA'].shape[1],-1)

        distances = torch.sum(
            (targetimg['defeatTDA'].unsqueeze(-1) - multiCenter.unsqueeze(0).unsqueeze(2).unsqueeze(3)) ** 2,
            dim=1)  # ([14, 256, 256, 10])
        # print('distances', distances.shape)
        nearest_centroids = torch.argmin(distances, dim=-1, keepdim=True)  # ([14, 256, 256, 1])
        # print('nearest_centroids', nearest_centroids.shape)
        pseudoMulticenter = torch.where(nearest_centroids.squeeze(-1) < targetimg['UnchgN'], self.zeroArray,
                                        self.oneArray).unsqueeze(1)  # [14, 1, 256, 256])
        # print('pseudoMulticenter', pseudoMulticenter.shape)
        mean_entropy = nn.functional.conv2d(outputEntropy, self.MeanKernal.repeat(c, 1, 1, 1),
                                            padding=self.kernelsize // 2, groups=c)
        # print('pseudoMulticenter',pseudoMulticenter.shape)
        outPesudo=self.erode_dilate_opencv(mask= pseudoMulticenter.float())#[14, 1, 256, 256]targetimg['Pseudo']
        outPesudo = self.erode_dilate_opencv(mask=targetimg['Pseudo'].float())
        # print('outPesudo',outPesudo.shape)
        if targetimg['MEntropyunchgT'] is not None and targetimg['MEntropychgT'] is not None:
            unchgRegion=torch.where(mean_entropy[:,0,:,:]<targetimg['MEntropyunchgT'], self.oneArray, self.zeroArray)
            chgRegion=torch.where(mean_entropy[:,1,:,:]<targetimg['MEntropychgT'], self.oneArray, self.zeroArray)
            # intersection = (unchgRegion * chgRegion)
            union = (torch.clamp(unchgRegion + chgRegion, 0, 1)).squeeze(1)
            selectregion=union
            if targetimg['i']<50:
                self.plotLabel(targetimg=targetimg, pesudoCenter=pseudoMulticenter,pesudoCenter2=outPesudo,outputEntropy=outputEntropy,
                           mean_entropy=mean_entropy,selectRegion=selectregion,Pro=Pro)
        else:
            selectregion=self.oneArray
            # self.plotLabel(targetimg=targetimg, pesudoCenter=pseudoMulticenter, outputEntropy=outputEntropy,
            #                mean_entropy=mean_entropy, selectRegion=selectregion)
        # print('mean_entropy', mean_entropy.shape)
        # self.plot(targetimg=targetimg, output_maskOut=outPesudo,
        #           softmaxOut=[Pro,mean_prob,var_prob,outputEntropy,mean_entropy,var_entropy], maskbefore=maskbefor)
        #
        return outPesudo,outputEntropy,mean_entropy,selectregion
class genpatchwithMaskProMid(nn.Module):
    def __init__(self,orisize=256,device=None,kernel=32):
        super(genpatchwithMaskProMid, self).__init__()
        self.kernel = kernel
        self.device = device
        self.zero = torch.tensor(0).to(self.device)
        self.stride = 1
        self.padding = 0
        self.orisize=orisize
        self.poolSize = (orisize - self.kernel) // self.stride + 1
        self.max = torch.tensor(orisize - 1).to(self.device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(self.poolSize).cuda(), torch.arange(self.poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool = nn.AvgPool2d(kernel_size=(self.kernel, self.kernel), stride=self.stride, padding=self.padding)
        self.poolSize = torch.tensor(self.poolSize, dtype=torch.int32)

    def median_class_prob(self,prob_graph, labels,device):

        batch_size, num_classes, height, width = prob_graph.size()
        medianArray=np.ones((batch_size,num_classes))
        for i in range(batch_size):
            label_index_chg = torch.nonzero(labels[i].reshape(height * width))
            outputfchg=prob_graph[i,1,:,:].reshape(height * width)[label_index_chg]
            if outputfchg.shape[0]==0:
                medianArray[i, 1]=torch.tensor([1]).to(device)
            else:
                medianArray[i, 1], _ = torch.median(outputfchg[:,0],dim=0)  # [28, 2]

            label_index_unchg = torch.nonzero((1-labels[i]).reshape(height * width))
            outputfUnchg = prob_graph[i, 0, :, :].reshape(height * width)[label_index_unchg]
            if outputfUnchg.shape[0] == 0:
                medianArray[i,0] = torch.tensor([1]).to(device)
            else:
                medianArray[i, 0], _ = torch.median(outputfUnchg[:, 0], dim=0)  # [28, 2]

        return medianArray
    def avg_class_pro(self,prob,label,device):
        ProUnchg=prob[:,0,:,:]*(1-label)
        proChg=prob[:,0,:,:]*label
        ProUnchgSum=torch.sum(ProUnchg,dim=[1,2])
        ProChgSum=torch.sum(proChg,dim=[1,2])

        UnchgNum=torch.sum(1-label,dim=[1,2])
        ChgNum=torch.sum(label,dim=[1,2])
        ProUnchgAvg=ProUnchgSum/UnchgNum
        proChgAvg=ProChgSum/ChgNum
        proAvg=torch.cat([ProUnchgAvg.unsqueeze(1),proChgAvg.unsqueeze(1)],dim=1)
        # print('ProUnchgAvg',ProUnchgAvg.shape,proChgAvg.shape)
        return proAvg
    def forward(self, infeat, labelTpesudo, labelT, FeatureDA, device=None, k=2):
        # output = F.softmax(infeat[:,:,self.kernel:-self.kernel,self.kernel:-self.kernel], dim=1)
        output = F.softmax(infeat,dim=1)
        outputf=output.clone().detach()

        output_featurePatch = self.avgpool(output)
        selectmask = torch.ones_like(output_featurePatch)
        # median_probs=self.median_class_prob(outputf,labelTpesudo[:,0,:,:].detach(),device=device)
        avg_probs=self.avg_class_pro(outputf,labelTpesudo[:,0,:,:].detach(),device=device)
        # print('median_probs',median_probs)

        PatchDict = {'unchg': [], 'chg': []}
        k=self.orisize*self.orisize//(self.kernel*self.kernel)//2
        self.num=0
        # k=1
        for c in range(2):
            # output_entropyPatch=output_entropyPatchOri*labelTpesudoOneHot[:,:,:,c]
            for kk in range(k):
                # print(selectmask[:,c,self.kernel//2:-self.kernel//2+1,self.kernel//2:-self.kernel//2+1].shape,output_featurePatch[:,c,:,:].shape)
                output_featurePatchFilter = (output_featurePatch[:, c, :, :]) * (selectmask[:, c, :, :]).detach()#[28, 241, 241]

                # output_featurePatchFilter=(output_featurePatch[:,c,:,:])*selectmask[:,c,:,:]
                output_featurePatchum = torch.flatten(output_featurePatchFilter.unsqueeze(1), start_dim=2,end_dim=3)  # torch.Size([2, 1, 12769])
                # select the first three maximum value
                values, indices = torch.topk(output_featurePatchum, k=1, dim=2, largest=True,
                                             sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])
                # print('values',values.shape,values)
                for i in range(output.shape[0]):
                    j = 0
                    if values[i, :, j]<avg_probs[i,c]:
                        continue
                    self.num=self.num+1
                    pxy = self.grid_xy[:, :, indices[i, :, j]].permute((1, 2, 0))  # torch.Size([1, 1, 2])
                    px = pxy[0, 0, 0]
                    py = pxy[0, 0, 1]

                    oxp = torch.tensor([torch.maximum(self.zero, px - self.kernel // self.stride // 2),
                                        torch.minimum(self.poolSize, px + self.kernel // self.stride // 2)],
                                       dtype=torch.int32).to(device)
                    oyp = torch.tensor([torch.maximum(self.zero, py - self.kernel // self.stride // 2),
                                        torch.minimum(self.poolSize, py + self.kernel // self.stride // 2)],
                                       dtype=torch.int32).to(device)
                    selectmask[i, c, oyp[0]:oyp[1], oxp[0]:oxp[1]] = 0
                    ox = torch.tensor([torch.maximum(self.zero, px * self.stride - self.padding),
                                       torch.minimum(self.max, px * self.stride + self.kernel - 1 - self.padding)],
                                      dtype=torch.int32).to(device)
                    oy = torch.tensor([torch.maximum(self.zero, py * self.stride - self.padding),
                                       torch.minimum(self.max, py * self.stride + self.kernel - 1 - self.padding)],
                                      dtype=torch.int32).to(device)
                    # outputOut=output[i,c,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    # print('in',torch.mean(outputOut),px,py,c,output_featurePatch[i,c,pyy,pxx]) #Veri

                    FeaturepatchOri = infeat[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    patchFeat = FeatureDA[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    labelTTrue = labelT[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    labelTpesudopatch = labelTpesudo[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                    elementDict = {'ClassiferT': FeaturepatchOri, 'patchFeatDA': patchFeat, 'provalue': values[i, :, j],
                                   'pointXY': oxy, 'c': c, 'i': i,
                                   'labelpesudo': labelTpesudopatch, 'labelTTrue': labelTTrue}
                    if FeaturepatchOri.shape[2] == self.kernel and FeaturepatchOri.shape[3] == self.kernel:
                        if c == 0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                    else:
                        print(FeaturepatchOri.shape)
                        print('genpatchError!!')
                        continue
                    # pxx = torch.tensor(pxy[0, 0, 0], dtype=torch.int32).to(device)
                    # pyy = torch.tensor(pxy[0, 0, 1], dtype=torch.int32).to(device)

                    # if output_featurePatch[i,c,pyy,pxx]<0.5:
                    #     print('values',c,kk,output_featurePatch[i,c,pyy,pxx])
        return PatchDict
    def Patch_select_gen(self,PatchDict,device):
        feat = []
        labelT = []
        labelpesudo=[]
        classiferT=[]
        for dict in PatchDict['unchg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        for dict in PatchDict['chg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        feat = torch.cat(feat, dim=0).to(device)
        labelT = torch.cat(labelT, dim=0).to(device)
        labelpesudo=torch.cat(labelpesudo, dim=0).to(device)
        classiferT=torch.cat(classiferT, dim=0).to(device)

        return feat,labelT,labelpesudo,classiferT

class genpatchwithMask2(nn.Module):
    def __init__(self, orisize=256, device=None, kernel=32):
        super(genpatchwithMask2, self).__init__()
        self.kernel = kernel
        self.device = device
        self.zero = torch.tensor(0).to(self.device)
        self.stride = 1
        self.padding = 0

        self.poolSize = (orisize - self.kernel) // self.stride + 1
        self.max = torch.tensor(orisize - 1).to(self.device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(self.poolSize).cuda(), torch.arange(self.poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool = nn.AvgPool2d(kernel_size=(self.kernel, self.kernel), stride=self.stride,
                                    padding=self.padding)
        self.poolSize = torch.tensor(self.poolSize, dtype=torch.int32)

    def forward(self, infeat, labelTpesudo, labelT, FeatureDA, device=None, k=2):
        # output = F.softmax(infeat[:,:,self.kernel:-self.kernel,self.kernel:-self.kernel], dim=1)
        # print('labelTpesudo',labelTpesudo.shape)
        output = F.softmax(infeat)
        # print(output_entropyPatch.shape,output.shape)#[20, 2, 256, 256]

        # labelTpesudoOneHot=F.one_hot(labelTpesudo.squeeze(1),num_classes=2)
        selectmask = torch.ones_like(output)
        # print('out',output_featurePatch.shape,selectmask.shape)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict = {'unchg': [], 'chg': []}
        # print('output_entropyPatchOri',output_entropyPatchOri.shape)
        # k=1
        for c in range(2):
            # output_entropyPatch=output_entropyPatchOri*labelTpesudoOneHot[:,:,:,c]
            for kk in range(k):
                # print(selectmask[:,c,self.kernel//2:-self.kernel//2+1,self.kernel//2:-self.kernel//2+1].shape,output_featurePatch[:,c,:,:].shape)
                output_featurePatchFilter = (output[:, c, :, :]) * (selectmask[:, c, :, :]).detach()
                output_featurePatch = self.avgpool(output_featurePatchFilter)

                # output_featurePatchFilter=(output_featurePatch[:,c,:,:])*selectmask[:,c,:,:]
                output_featurePatchum = torch.flatten(output_featurePatch.unsqueeze(1), start_dim=2,
                                                      end_dim=3)  # torch.Size([2, 1, 12769])
                # select the first three maximum value
                values, indices = torch.topk(output_featurePatchum, k=1, dim=2, largest=True,
                                             sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])
                for i in range(output.shape[0]):
                    j = 0
                    pxy = self.grid_xy[:, :, indices[i, :, j]].permute((1, 2, 0))  # torch.Size([1, 1, 2])
                    px = pxy[0, 0, 0]
                    py = pxy[0, 0, 1]

                    # oxp = torch.tensor([torch.maximum(self.zero, px - self.kernel // self.stride // 2),
                    #                     torch.minimum(self.poolSize, px + self.kernel // self.stride // 2)],
                    #                    dtype=torch.int32).to(device)
                    # oyp = torch.tensor([torch.maximum(self.zero, py - self.kernel // self.stride // 2),
                    #                     torch.minimum(self.poolSize, py + self.kernel // self.stride // 2)],
                    #                    dtype=torch.int32).to(device)
                    #
                    ox = torch.tensor([torch.maximum(self.zero, px * self.stride - self.padding),
                                       torch.minimum(self.max, px * self.stride + self.kernel - 1 - self.padding)],
                                      dtype=torch.int32).to(device)
                    oy = torch.tensor([torch.maximum(self.zero, py * self.stride - self.padding),
                                       torch.minimum(self.max, py * self.stride + self.kernel - 1 - self.padding)],
                                      dtype=torch.int32).to(device)
                    selectmask[i, c, ox[0]:oy[1], ox[0]:oy[1]] = 0
                    # outputOut=output[i,c,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    # print('in',torch.mean(outputOut),px,py,c,output_featurePatch[i,c,pyy,pxx]) #Veri

                    FeaturepatchOri = infeat[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    patchFeat = FeatureDA[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    labelTTrue = labelT[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    labelTpesudopatch = labelTpesudo[i, :, oy[0]:oy[1] + 1, ox[0]:ox[1] + 1].unsqueeze(0)
                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                    elementDict = {'ClassiferT': FeaturepatchOri, 'patchFeatDA': patchFeat,
                                   'provalue': values[i, :, j],
                                   'pointXY': oxy, 'c': c, 'i': j,
                                   'labelpesudo': labelTpesudopatch, 'labelTTrue': labelTTrue}
                    if FeaturepatchOri.shape[2] == self.kernel and FeaturepatchOri.shape[3] == self.kernel:
                        if c == 0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                    else:
                        print(FeaturepatchOri.shape)
                        print('genpatchError!!')
                        continue
                    # pxx = torch.tensor(pxy[0, 0, 0], dtype=torch.int32).to(device)
                    # pyy = torch.tensor(pxy[0, 0, 1], dtype=torch.int32).to(device)

                    # if output_featurePatch[i,c,pyy,pxx]<0.5:
                    #     print('values',c,kk,output_featurePatch[i,c,pyy,pxx])
        return PatchDict
    def Patch_select_gen(self,PatchDict,device):
        feat = []
        labelT = []
        labelpesudo=[]
        classiferT=[]
        for dict in PatchDict['unchg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        for dict in PatchDict['chg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        feat = torch.cat(feat, dim=0).to(device)
        labelT = torch.cat(labelT, dim=0).to(device)
        labelpesudo=torch.cat(labelpesudo, dim=0).to(device)
        classiferT=torch.cat(classiferT, dim=0).to(device)

        return feat,labelT,labelpesudo,classiferT
class genpatch(nn.Module):
    def __init__(self,orisize=256,device=None,kernel=32):
        super(genpatch, self).__init__()
        self.kernel = kernel
        self.zero = torch.tensor(0).to(device)
        self.stride=2
        self.padding = 0

        poolSize=(orisize-self.kernel)//self.stride+1
        print('poolSize',poolSize)
        self.max = torch.tensor(orisize - 1).to(device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(poolSize).cuda(), torch.arange(poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        self.avgpool=nn.AvgPool2d(kernel_size=(self.kernel,self.kernel),stride=self.stride,padding=self.padding)
        # self.kernelarray=nn.Parameter(torch.ones((1,2,self.kernel,self.kernel)),requires_grad=False)
        # self.convAvg=F.conv2d()
    def forward(self, infeat, labelTpesudo,labelT,FeatureDA, device=None,k=2):
        # output = F.softmax(infeat[:,:,self.kernel:-self.kernel,self.kernel:-self.kernel], dim=1)
        output=F.softmax(infeat)
        output_featurePatch = self.avgpool(output)
        # output_featurePatch=F.conv2d(output,self.kernelarray,stride=self.stride,padding=self.padding)
        print('infeat',infeat.shape,output_featurePatch.shape)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict= {'unchg':[],'chg':[]}
        # print('output_featurePatch',output_featurePatch.shape)
        # k=1
        for c in range(2):
            output_featurePatchum = torch.flatten(output_featurePatch[:, c, :, :].unsqueeze(1), start_dim=2,
                                                       end_dim=3)  # torch.Size([2, 1, 12769])
            values, indices = torch.topk(output_featurePatchum, k=k, dim=2, largest=True,
                                                   sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])#select the first three maximum value
            # print('values',values)
            for i in range(output.shape[0]):
                for j in range(k):
                    pxy=self.grid_xy[:,:,indices[i,:,j]].permute((1,2,0))# torch.Size([1, 1, 2])
                    px=pxy[0,0,0]
                    py=pxy[0,0,1]
                    # pxx = torch.tensor(pxy[0, 0, 0],dtype=torch.int32).to(device)
                    # pyy = torch.tensor(pxy[0, 0, 1],dtype=torch.int32).to(device)
                    # print()
                    ox=torch.tensor([torch.maximum(self.zero,px*self.stride-self.padding),
                                     torch.minimum(self.max,px*self.stride+self.kernel-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    oy=torch.tensor([torch.maximum(self.zero,py*self.stride-self.padding),
                                     torch.minimum(self.max,py*self.stride+self.kernel-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    # outputOut=output[i,c,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    # print('in',torch.mean(outputOut),px,py,c,output_featurePatch[i,c,pyy,pxx]) #Veri

                    FeaturepatchOri=infeat[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    patchFeat=FeatureDA[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    labelTTrue=labelT[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    labelTpesudopatch=labelTpesudo[i,:,oy[0]:oy[1]+1,ox[0]:ox[1]+1].unsqueeze(0)
                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                    elementDict = {'ClassiferT':FeaturepatchOri,'patchFeatDA':patchFeat,'provalue':values[i,:,j],'pointXY':oxy,'c':c,'i':j,
                                   'labelpesudo':labelTpesudopatch,'labelTTrue':labelTTrue}
                    if FeaturepatchOri.shape[2]==self.kernel and FeaturepatchOri.shape[3]==self.kernel and j<k:
                        if c==0:
                            PatchDict['unchg'].append(elementDict)
                        else:
                            PatchDict['chg'].append(elementDict)
                        break
                    else:
                        print(FeaturepatchOri.shape)
                        print('genpatchError!!')
                        continue


        return PatchDict
    def Patch_select_gen(self,PatchDict,device):
        feat = []
        labelT = []
        labelpesudo=[]
        classiferT=[]
        for dict in PatchDict['unchg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        for dict in PatchDict['chg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        feat = torch.cat(feat, dim=0).to(device)
        labelT = torch.cat(labelT, dim=0).to(device)
        labelpesudo=torch.cat(labelpesudo, dim=0).to(device)
        classiferT=torch.cat(classiferT, dim=0).to(device)

        return feat,labelT,labelpesudo,classiferT
class genpixel(nn.Module):
    def __init__(self,orisize=256,device=None):
        super(genpixel, self).__init__()
        self.kernel = 1
        self.zero = torch.tensor(0).to(device)
        self.stride=1
        self.padding = 0
        poolSize=(orisize-self.kernel)//self.stride+1
        self.max = torch.tensor(orisize - 1).to(device)
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(poolSize).cuda(), torch.arange(poolSize).cuda()])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).permute((2, 0, 1)).unsqueeze(
            1).float()  # torch.Size([113, 113, 2])
        self.grid_xy = torch.flatten(grid_xy, start_dim=2, end_dim=3)
        # self.avgpool=nn.AvgPool2d(kernel_size=(self.kernel,self.kernel),stride=self.stride,padding=self.padding)
    def forward(self, infeat, labelTpesudo,labelT,FeatureDA, device=None,k=2,size=32):
        # output = F.softmax(infeat[:,:,self.kernel:-self.kernel,self.kernel:-self.kernel], dim=1)
        output=F.softmax(infeat)
        # output_featurePatch = self.avgpool(output)
        # proPatch=torch.where(output_featurePatch>0.9,ones,zeros).cuda()
        # output_featurePatch.
        # x = F.unfold(output_featurePatch, kernel_size=2, dilation=1, stride=2)#torch.Size([28, 2, 113, 113])
        PatchDict= {'unchg':[],'chg':[]}

        for c in range(2):
            output_featurePatchum = torch.flatten(output[:, c, :, :].unsqueeze(1), start_dim=2,
                                                       end_dim=3)  # torch.Size([2, 1, 12769])
            values, indices = torch.topk(output_featurePatchum, k=k*size*size, dim=2, largest=True,
                                                   sorted=True)  # torch.Size([2, 1, 2]) torch.Size([2, 1, 2])#select the first three maximum value
            # print('values',values)
            for i in range(output.shape[0]):
                for j in range(k):
                    pxy=self.grid_xy[:,:,indices[i,:,j]].permute((1,2,0))# torch.Size([1, 1, 2])
                    px=pxy[0,0,0]
                    py=pxy[0,0,1]
                    ox=torch.tensor([torch.maximum(self.zero,px*self.stride-self.padding),
                                     torch.minimum(self.max,px*self.stride+32-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    oy=torch.tensor([torch.maximum(self.zero,py*self.stride-self.padding),
                                     torch.minimum(self.max,py*self.stride+32-1-self.padding)],
                                    dtype=torch.int32).to(device)
                    outputOut=output[i,:,oy[0],ox[0]].unsqueeze(0)
                    # print('in',outputOut,px,py,c)
                    FeaturepatchOri=infeat[i,:,oy[0],ox[0]].unsqueeze(0)
                    patchFeat=FeatureDA[i,:,oy[0],ox[0]].unsqueeze(0)
                    labelTTrue=labelT[i,:,oy[0],ox[0]].unsqueeze(0)
                    labelTpesudopatch=labelTpesudo[i,:,oy[0],ox[0]].unsqueeze(0)


                    oxy = torch.cat([ox.unsqueeze(0), oy.unsqueeze(0)], dim=0)  # torch.Size([2, 2])
                    elementDict = {'ClassiferT':FeaturepatchOri,'patchFeatDA':patchFeat,'provalue':values[i,:,j],'pointXY':oxy,'c':c,'i':j,
                                   'labelpesudo':labelTpesudopatch,'labelTTrue':labelTTrue}
                    if c == 0:
                        PatchDict['unchg'].append(elementDict)
                    else:
                        PatchDict['chg'].append(elementDict)


        return PatchDict

    def Pixel_select_gen(self, PatchDict, device):
        feat = []
        labelT = []
        labelpesudo = []
        classiferT = []
        for dict in PatchDict['unchg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        for dict in PatchDict['chg']:
            feat.append(dict['patchFeatDA'])
            labelT.append(dict['labelTTrue'])
            labelpesudo.append(dict['labelpesudo'])
            classiferT.append(dict['ClassiferT'])
        feat = torch.cat(feat, dim=0).to(device)
        labelT = torch.cat(labelT, dim=0).to(device)
        labelpesudo = torch.cat(labelpesudo, dim=0).to(device)
        classiferT = torch.cat(classiferT, dim=0).to(device)
        # print('classiferT',classiferT.shape)
        return feat, labelT, labelpesudo, classiferT


class CenterTOpEXnewMultiC(nn.Module):
    def __init__(self, DEVICE, dist_type='cos'):
        super(CenterTOpEXnewMultiC, self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes = 2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)
    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def assign_labels(self, feats, filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labelsori = torch.min(dists, dim=1)
        if filter:
            zeros=torch.zeros_like(labelsori)
            ones=torch.ones_like(labelsori)
            labels=torch.where(labelsori>self.unchgCenterNum-1,ones,zeros)#when multiCenter use
        return dists, labels,labelsori

    def selecdata(self, feature, label):
        # label 6, 1, 128, 128
        # feature 6, 32, 128, 128
        label_flatten = label
        feature_flatten = feature
        label_index = torch.nonzero(label_flatten)
        label_index = label_index.squeeze(1)
        feature_flatten_select = feature_flatten[label_index]  # bs,c

        return feature_flatten_select, label_index
    def forward(self,FeatureT,centerInit,num1,num2,varflag=False,unchgN=1,chgN=1,iterC=False):
        self.unchgCenterNum=unchgN#if multi-center,the num of unchgCenter
        self.chgCenterNum=chgN#if multi-center,the num of chgCenter
        centersIter = None #the center of last iter
        Ci = 0  # num of iter
        bb,c,w,h=FeatureT.shape
        # FeatureTi = FeatureT
        FeatureT = FeatureT.reshape(bb, c, w*h)
        # print('b,c,w,h',b,c,w,h)
        # print('FeatureTi',FeatureT)
        centersIterout = 0
        labelsout = []
        labels_onehotout = []
        dist2centerT = []
        Cinidist = 0
        labelTinit = []
        for b in range(FeatureT.shape[0]):
            CinidistW=0.001
            while True:
                if iterC:#Whether to iterate the center
                    if Ci == 0 and centersIter is None:#first time
                        self.centers = centerInit
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # init feature
                    elif Ci == 0 and centersIter is not None:
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                        self.centers = self.centers + CinidistW * CinidistW * (centersIter - self.centers)  # weighted
                    else:
                        self.centers = self.centers + CinidistW * CinidistW * (centersIter - self.centers)
                        if Ci > 2:#
                            labelsout.append(labels)
                            labels_onehotout.append(labels_onehot.unsqueeze(0))
                            dist2centerT.append(dist2center.unsqueeze(0))
                            centersIterout = centersIterout + centersIter
                            Ci = 0
                            break
                else:
                    if Ci == 0 and centersIter is not None:
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                        # self.centers=self.centers+0.1*(centersIter-self.centers)
                        self.centers = centerInit
                    elif Ci == 0 and centersIter is None:
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                        self.centers = centerInit
                    else:
                        self.centers = self.centers + 0.001 * (centersIter - self.centers)
                        if Ci > 0:
                            labelsout.append(labels)
                            labels_onehotout.append(labels_onehot.unsqueeze(0))
                            dist2centerT.append(dist2center.unsqueeze(0))
                            Ci = 0
                            break

                dist2center, labels, labelsori = self.assign_labels(FeatureTb, filter=True)  # [65536, 2] [65536] [65536, 32]
                dist2center = torch.cat([dist2center[:, 0:self.unchgCenterNum].mean(1).unsqueeze(1),
                                         dist2center[:, self.unchgCenterNum:].mean(1).unsqueeze(1)], dim=1)

                labels_onehot = self.to_onehot(labels, self.num_classes)  # [65536, 2]
                labelOriOnehot = torch.zeros(labelsori.shape[0], self.unchgCenterNum + self.chgCenterNum).to(
                            self.device).scatter_(1, labelsori.unsqueeze(1), 1)  # ([65536, 6])
                #generate target center by pseudo label
                FeatureTbFilter = FeatureTb.unsqueeze(2)*labelOriOnehot.unsqueeze(1)
                Num = labelOriOnehot.sum(0)+1
                FeatureTbFilter = FeatureTbFilter.sum(0)/Num.unsqueeze(0)#[32, 6]
                # unchgSelectFeat
                centersIter = FeatureTbFilter.permute(1,0)
                if Ci==0:
                    labels = labels.unsqueeze(0)  # 1, 65536 only be selected by centerDist not filter(var)
                    labelTinit.append(labels)

                Ci = Ci + 1
        centersIterout = self.centers
        if True:
            labelsout=torch.cat(labelsout,dim=0).reshape(bb,1,w,h)
            labelTinit=torch.cat(labelTinit,dim=0).reshape(bb,1,w,h)#input pseudo label(fristly get)
            # print('labelTinit',labelTinit.shape)
            labels_onehotout=torch.cat(labels_onehotout,dim=0).reshape(bb,2,w,h)#[13, 65536, 2]
            dist2centerTout=torch.cat(dist2centerT,dim=0).reshape(bb,2,w,h)#[13, 65536, 2]
        else:
            labelsout = torch.cat(labelsout, dim=0)
            labelTinit = torch.cat(labelTinit, dim=0)  # input pseudo label(fristly get)
            # print('labelTinit',labelTinit.shape)
            labels_onehotout = torch.cat(labels_onehotout, dim=0)  # [13, 65536, 2]
            dist2centerTout = torch.cat(dist2centerT, dim=0)  # [13, 65536, 2]
        # dist2centerT=(dist2centerTori-dist2centerTori.min(1)[0].unsqueeze(1))/(dist2centerTori.max(1)[0].unsqueeze(1)-dist2centerTori.min(1)[0].unsqueeze(1)+0.0000001)
        # dist2centerT=(1-dist2centerT)
        # Weight=dist2centerT
        # Cinidist=Cinidist.sum()/FeatureT.shape[0]#total center distance
        # print('centersIterout',centersIterout.shape)
        return centersIterout.detach(),[labelsout,labels_onehotout,dist2centerTout,labelTinit]




class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        # print('pointA',pointA.shape, pointB.shape)
        return getattr(self, self.dist_type)(
		pointA, pointB, cross)

    def L2(self, pointA, pointB, cross):
        # dist = F.pairwise_distance(pointA, pointB)
        dist = torch.pow(pointA-pointB,2)
        return dist
    def cos(self, pointA, pointB, cross):
        # print('pointA',pointA.sum())
        # pointA = F.normalize(pointA, dim=1)
        # pointB = F.normalize(pointB, dim=1)
        if not cross:
            # print(pointA.shape,pointB.shape)
            dist = F.cosine_similarity(pointA,pointB, dim=1)
            # return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
            return dist
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
            # return F.cosine_similarity(pointA,pointB, dim=1)

class PatchLoss():
    def patchLoss_select(self, PatchDict, lossF, device):
        lossT = 0
        # for dict in PatchDict['unchg']:
        #     # print(dict['patch'].shape,dict['label'].shape)
        #     loss=lossF(dict['patch'],dict['label'].long())
        #     lossT=lossT+loss
        # for dict in PatchDict['chg']:
        #     loss = lossF(dict['patch'], dict['label'].long())
        #     lossT = lossT + loss
        feat = []
        label = []
        for dict in PatchDict['chg']:
            feat.append(dict['patch'])
            label.append(dict['label'])
        feat = torch.cat(feat, dim=0).to(device)
        label = torch.cat(label, dim=0).to(device)
        lossT = lossF(feat, label.long())
        return lossT
    def patchLoss_Paste(self,coor,label,feature,device,lossF,size=64):
        labelP= label[:, :, coor[0]:coor[0] + size, coor[1]:coor[1] + size].to(device)
        featureP=feature[:, :, coor[0]:coor[0] + size, coor[1]:coor[1] + size].to(device)
        loss = lossF(featureP, labelP.long())

        return loss
class SelecFeatNew():
    def __init__(self,device,chgthreshold = 1200,unchgthreshold = 1200,dist_type='cos'):
        self.device=device
        self.chgthreshold = chgthreshold
        self.unchgthreshold = unchgthreshold
        self.Dist = DIST(dist_type)

    def select_featureTAll(self, target, pseudo_label, softmaxLabel, p=0, pe=0):

        pseudo_label = pseudo_label.unsqueeze(1)
        # softmaxLabelori=softmaxLabel.reshape(-1,2,s_label.shape[2],s_label.shape[3])#[bs,2,h,w]->[bs,h,w,2]
        softmaxLabelori = softmaxLabel
        self.uu = (softmaxLabel[:, 0, :, :] * (1 - pseudo_label.squeeze(1))).sum() / ((1 - pseudo_label).sum() + 1)
        self.cc = ((softmaxLabel[:, 1, :, :]) * pseudo_label.squeeze(1)).sum() / (pseudo_label.sum() + 1)
        softmaxLabel = torch.flatten(softmaxLabelori.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2

        ######################change
        ones = torch.ones_like(pseudo_label)
        zeros = torch.zeros_like(pseudo_label)

        pseudo_labeltChg = torch.where(softmaxLabelori[:, 1, :, :].unsqueeze(1) > (p - pe), pseudo_label,
                                       zeros).detach()
        ############################3############################3############################3############################3############################3
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdataNoRam(target, pseudo_labeltChg)

        target_chg_flatten_select = target_chg_flatten[target_chg_index , :]  # bs,c
        softmaxLabel_chg_select = softmaxLabel[target_chg_index]  # [bs,2]

        ####################unchg
        pseudo_labeltunChg = torch.where(softmaxLabelori[:, 0, :, :].unsqueeze(1) > p, pseudo_label, ones).detach()
        ############################3############################3############################3############################3
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdataNoRam(target,
                                                                                               1 - pseudo_labeltunChg)

        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index, :]  # bs,c
        softmaxLabel_unchg_select = softmaxLabel[target_unchg_index]
        # print(target_unchg_flatten_select.shape,target_chg_flatten_select.shape)
        self.chgNum = target_unchg_flatten_select.shape[0]
        self.unchgNum = target_chg_flatten_select.shape[0]
        unchglabel = self.to_onehot(torch.zeros_like(softmaxLabel_unchg_select[:, 0]).long(), 2)
        chglabel = self.to_onehot(torch.ones_like(softmaxLabel_unchg_select[:, 1]).long(), 2)
        # print(unchglabel, chglabel)
        # s_label_select = torch.cat([unchglabel, chglabel], dim=0).detach()

        t_label_select = torch.cat([softmaxLabel_unchg_select, softmaxLabel_chg_select], dim=0).detach()
        t_label_pseudo = torch.cat([unchglabel, chglabel], dim=0).detach()

        return target_chg_flatten_select, target_unchg_flatten_select, t_label_select, t_label_pseudo
    def select_featureTRam(self, target, pseudo_label, softmaxLabel, p=0, pe=0):
        pseudo_label = pseudo_label.unsqueeze(1)
        # softmaxLabelori=softmaxLabel.reshape(-1,2,s_label.shape[2],s_label.shape[3])#[bs,2,h,w]->[bs,h,w,2]
        softmaxLabelori = softmaxLabel
        self.uu = (softmaxLabel[:, 0, :, :] * (1 - pseudo_label.squeeze(1))).sum() / ((1 - pseudo_label).sum() + 1)
        self.cc = ((softmaxLabel[:, 1, :, :]) * pseudo_label.squeeze(1)).sum() / (pseudo_label.sum() + 1)
        softmaxLabel = torch.flatten(softmaxLabelori.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2
        ######################change
        ones = torch.ones_like(pseudo_label)
        zeros = torch.zeros_like(pseudo_label)

        pseudo_labeltChg = torch.where(softmaxLabelori[:, 1, :, :].unsqueeze(1) > (p - pe), pseudo_label,zeros).detach()
        ############################3############################3############################3############################3############################3
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pseudo_labeltChg)
        ####################unchg
        pseudo_labeltunChg = torch.where(softmaxLabelori[:, 0, :, :].unsqueeze(1) > p, pseudo_label,ones).detach()
        ############################3############################3############################3############################3
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target,
                                                                                               1 - pseudo_labeltunChg)
        # print('be',target_unchg_flatten_select.shape,target_chg_flatten_select.shape)
        if target_unchg_index.shape[0] > target_chg_index.shape[0]:
            unchgthreshold = target_chg_index.shape[0]
            # print('target_unchg_index',target_unchg_index.shape,unchgthreshold)
            target_unchg_flatten_select = target_unchg_flatten_select[0:unchgthreshold, :]  # bs,c
        elif target_unchg_index.shape[0] < target_chg_index.shape[0]:
            chgthreshold = target_unchg_index.shape[0]
            target_chg_flatten_select = target_chg_flatten_select[0:chgthreshold, :]  # bs,c
        # print('target_chg_flatten_select',target_chg_flatten_select)
        # print('af',target_unchg_flatten_select.shape,target_chg_flatten_select.shape)
        softmaxLabel_chg_select = softmaxLabel[target_chg_index]  # [bs,2]
        softmaxLabel_unchg_select = softmaxLabel[target_unchg_index]
        # self.chgNum = target_unchg_flatten_select.shape[0]
        # self.unchgNum = target_chg_flatten_select.shape[0]
        unchglabel = self.to_onehot(torch.zeros_like(softmaxLabel_unchg_select[:, 0]).long(), 2)
        chglabel = self.to_onehot(torch.ones_like(softmaxLabel_unchg_select[:, 1]).long(), 2)
        # print(unchglabel, chglabel)
        # s_label_select = torch.cat([unchglabel, chglabel], dim=0).detach()
        t_label_select = torch.cat([softmaxLabel_unchg_select, softmaxLabel_chg_select], dim=0).detach()
        t_label_pseudo = torch.cat([unchglabel, chglabel], dim=0).detach()
        return target_chg_flatten_select, target_unchg_flatten_select, t_label_select, t_label_pseudo
    def selecdataRam(self, feature, label):
        # label 6, 1, 128, 128
        # feature 6, 32, 128, 128

        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)
        label_index = torch.nonzero(label_flatten)
        label_index = torch.flatten(label_index)
        # print('label_index',label_index.shape,label.sum())
        # print('label_index.nelement()',label_index.nelement(),label_index_rand)
        feature_flatten_select = feature_flatten[label_index, :]  # bs,c
        # print('feature_flatten_select.nelement()', feature_flatten_select.shape,label_flatten.sum())
        return feature_flatten_select, label_index, feature_flatten
    def selecdataNoRam(self, feature, label):
        # label 6, 1, 128, 128
        # feature 6, 32, 128, 128

        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)
        label_index = torch.nonzero(label_flatten)
        label_index = torch.flatten(label_index)
        # print('label_index',label_index.shape,label.sum())
        # print('label_index.nelement()',label_index.nelement(),label_index_rand)
        feature_flatten_select = feature_flatten[label_index, :]  # bs,c
        # print('feature_flatten_select.nelement()', feature_flatten_select.shape,label_flatten.sum())
        return feature_flatten_select, label_index, feature_flatten
    def selecdata(self, feature, label):
        # label 6, 1, 128, 128
        # feature 6, 32, 128, 128

        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)
        label_index = torch.nonzero(label_flatten)
        label_index = torch.flatten(label_index)
        # print('label_index',label_index.shape,label.sum())
        label_index_rand = torch.randperm(label_index.nelement())
        # print('label_index.nelement()',label_index.nelement(),label_index_rand)
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[label_index, :]  # bs,c
        # print('feature_flatten_select.nelement()', feature_flatten_select.shape,label_flatten.sum())
        return feature_flatten_select, label_index, feature_flatten

    def to_onehot(self, label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot

    def select_featureS(self, source, s_label):
        chgthreshold = 2400  # select 1000 pixel
        unchgthreshold = 2400
        self.chgthreshold = chgthreshold
        self.unchgthreshold = unchgthreshold
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, s_label)
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - s_label)
        # source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold], :]  # bs,c
        # source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        source_unchg_flatten_select = source_unchg_flatten_select[0:unchgthreshold]  # bs,c
        source_chg_flatten_select = source_chg_flatten_select[0:chgthreshold]  # bs,c
        return source_chg_flatten_select, source_unchg_flatten_select

    def position(self, device):
        h = 256
        w = 256
        xx = torch.arange(h)
        yy = torch.arange(w)
        x_expand = xx.unsqueeze(1).expand(5, h, w).unsqueeze(-1)
        y_expand = yy.unsqueeze(0).expand(5, h, w).unsqueeze(-1)
        aa = torch.ones((h, w)).unsqueeze(0).unsqueeze(-1)
        cc = torch.cat([aa, aa * 2, aa * 3, aa * 4, aa * 5], dim=0)
        p = torch.cat([x_expand, y_expand, cc], dim=-1)

        p = torch.flatten(p, start_dim=0, end_dim=2)
        return p
    def select_featureT(self, target, pseudo_label, softmaxLabel,  p=0, pe=0):

        pseudo_label = pseudo_label.unsqueeze(1)
        # softmaxLabelori=softmaxLabel.reshape(-1,2,s_label.shape[2],s_label.shape[3])#[bs,2,h,w]->[bs,h,w,2]
        softmaxLabelori = softmaxLabel
        self.uu = (softmaxLabel[:, 0, :, :] * (1 - pseudo_label.squeeze(1))).sum() / ((1 - pseudo_label).sum() + 1)
        self.cc = ((softmaxLabel[:, 1, :, :]) * pseudo_label.squeeze(1)).sum() / (pseudo_label.sum() + 1)
        softmaxLabel = torch.flatten(softmaxLabelori.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2

        ######################change
        ones = torch.ones_like(pseudo_label)
        zeros = torch.zeros_like(pseudo_label)

        pseudo_labeltChg = torch.where(softmaxLabelori[:, 1, :, :].unsqueeze(1) > (p - pe), pseudo_label, zeros).detach()
        ############################3############################3############################3############################3############################3
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pseudo_labeltChg)

        if target_chg_index.shape[0] < self.chgthreshold:
            chgthreshold = target_chg_index.shape[0]
        else:
            chgthreshold=self.chgthreshold
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold], :]  # bs,c
        softmaxLabel_chg_select = softmaxLabel[target_chg_index[0:chgthreshold]]  # [bs,2]

        ####################unchg
        pseudo_labeltunChg = torch.where(softmaxLabelori[:, 0, :, :].unsqueeze(1) > p, pseudo_label,
                                         ones).detach()  ############################3############################3############################3############################3

        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target,1 - pseudo_labeltunChg)

        if  target_unchg_index.shape[0] < self.unchgthreshold:
            unchgthreshold = target_unchg_index.shape[0]
        else:
            unchgthreshold=self.unchgthreshold
        if unchgthreshold > chgthreshold:
            unchgthreshold = chgthreshold
        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold], :]  # bs,c
        softmaxLabel_unchg_select = softmaxLabel[target_unchg_index[0:unchgthreshold]]
        self.chgNum = chgthreshold
        self.unchgNum = unchgthreshold
        unchglabel = self.to_onehot(torch.zeros_like(softmaxLabel_unchg_select[:, 0]).long(), 2)
        chglabel = self.to_onehot(torch.ones_like(softmaxLabel_unchg_select[:, 1]).long(), 2)
        # print(unchglabel, chglabel)
        # s_label_select = torch.cat([unchglabel, chglabel], dim=0).detach()

        t_label_select = torch.cat([softmaxLabel_unchg_select, softmaxLabel_chg_select], dim=0).detach()
        t_label_pseudo= torch.cat([unchglabel, chglabel], dim=0).detach()

        return target_chg_flatten_select, target_unchg_flatten_select, t_label_select, t_label_pseudo
    def select_featureST(self, source, s_label, target, pseudo_label, softmaxLabel, softLog, p=0, pe=0):


        pseudo_label = pseudo_label.unsqueeze(1)
        # print('softmaxLabel',softmaxLabel.shape)#[13, 2, 65536]
        # softmaxLabelori=softmaxLabel.reshape(-1,2,s_label.shape[2],s_label.shape[3])#[bs,2,h,w]->[bs,h,w,2]
        softmaxLabelori = softmaxLabel
        # print('softmaxLabelori',softmaxLabelori.shape,pseudo_label.shape)
        self.uu = (softmaxLabel[:, 0, :, :] * (1 - pseudo_label.squeeze(1))).sum() / ((1 - pseudo_label).sum() + 1)
        self.cc = ((softmaxLabel[:, 1, :, :]) * pseudo_label.squeeze(1)).sum() / (pseudo_label.sum() + 1)

        softmaxLabel = torch.flatten(softmaxLabelori.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2

        # softLogS = softLogS.reshape(-1, 2, s_label.shape[2], s_label.shape[3])  # [bs,2,h,w]->[bs,h,w,2]
        # softLogS = torch.flatten(softLogS.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2
        # softLogT = softLogT.reshape(-1, 2, s_label.shape[2], s_label.shape[3])  # [bs,2,h,w]->[bs,h,w,2]
        # softLogT = torch.flatten(softLogT.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2
        ######################change
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, s_label)
        ones = torch.ones_like(pseudo_label)
        zeros = torch.zeros_like(pseudo_label)

        pseudo_labeltChg = torch.where(softmaxLabelori[:, 1, :, :].unsqueeze(1) > (p - pe), pseudo_label,
                                       zeros).detach()  ############################3############################3############################3############################3############################3
        # pseudo_labeltChg=torch.where((softmaxLabelori[:,1,:,:]/softmaxLabelori[:,1,:,:].max()).unsqueeze(1)>p,pseudo_label,zeros)
        # print('pseudo_label',pseudo_labeltChg.shape,pseudo_label.shape,pseudo_label.sum(),pseudo_labeltChg.sum())
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pseudo_labeltChg)

        if source_chg_index.shape[0] < self.chgthreshold or target_chg_index.shape[0] < self.chgthreshold:
            chgthreshold = np.minimum(source_chg_index.shape[0], target_chg_index.shape[0])
        source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold], :]  # bs,c
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold], :]  # bs,c
        # print(pp.shape,target_chg_flatten.shape)
        # target_pchg=pp[target_chg_index[0:chgthreshold].cpu(),:].unsqueeze(0)
        # print('target_p',target_p)
        softmaxLabel_chg_select = softmaxLabel[target_chg_index[0:chgthreshold]]  # [bs,2]

        # softLogT_chg_select = softLogT[target_chg_index[0:chgthreshold]]  # [bs,2]
        # softLogS_chg_select = softLogS[source_chg_index[0:chgthreshold]]
        # print(softmaxLabel_chg_select)
        # print('softmaxLabel_chg_select',softmaxLabel_chg_select.shape)
        # target_chg_flatten_selectW = target_chg_flatten_select * softmaxLabel_chg_select[:, 1].unsqueeze(1)
        ####################unchg
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - s_label)
        # print('softmaxLabel',softmaxLabel.shape)
        pseudo_labeltunChg = torch.where(softmaxLabelori[:, 0, :, :].unsqueeze(1) > p, pseudo_label,
                                         ones).detach()  ############################3############################3############################3############################3
        # pseudo_labeltunChg = torch.where((softmaxLabelori[:,0,:,:]/softmaxLabelori[:,0,:,:].max()).unsqueeze(1)>p, pseudo_label, ones)

        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target,
                                                                                               1 - pseudo_labeltunChg)

        if source_unchg_index.shape[0] < self.unchgthreshold or target_unchg_index.shape[0] < self.unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        if unchgthreshold > chgthreshold:
            unchgthreshold = chgthreshold
        source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold], :]  # bs,c
        # softLogS_unchg_select=softLogS[source_unchg_index[0:unchgthreshold]]

        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold], :]  # bs,c
        # target_punchg = pp[target_unchg_index[0:unchgthreshold].cpu(), :].unsqueeze(0)
        softmaxLabel_unchg_select = softmaxLabel[target_unchg_index[0:unchgthreshold]]
        # softLogT_unchg_select = softLogT[target_unchg_index[0:unchgthreshold]]
        # print('target_pchg',target_pchg.shape,target_punchg.shape)
        # target_unchg_flatten_selectW = target_unchg_flatten_select * softmaxLabel_unchg_select[:, 0].unsqueeze(1)#weight
        self.chgNum = chgthreshold
        self.unchgNum = unchgthreshold
        unchglabel = self.to_onehot(torch.zeros_like(softmaxLabel_unchg_select[:, 0]).long(), 2)
        chglabel = self.to_onehot(torch.ones_like(softmaxLabel_unchg_select[:, 1]).long(), 2)
        # print(unchglabel, chglabel)
        # print('s',softmaxLabel_unchg_select.shape,softmaxLabel_chg_select[1].shape,softmaxLabel_unchg_select[:,0].min(),softmaxLabel_chg_select[:,1].min())
        s_label_select = torch.cat([unchglabel, chglabel], dim=0).detach()
        # print('s_label_select',s_label_select)
        t_label_select = torch.cat([softmaxLabel_unchg_select, softmaxLabel_chg_select], dim=0).detach()
        # print('softmaxLabel_unchg_select',t_label_select.shape)
        t_label_select2 = torch.cat([softmaxLabel_unchg_select, softmaxLabel_chg_select], dim=0).detach()
        # softLogS=torch.cat([softLogS_unchg_select,softLogS_chg_select], dim=0)
        # softLogT=torch.cat([softLogT_unchg_select,softLogT_chg_select], dim=0)

        # print('t_label_select2',t_label_select2.shape)
        return source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select, \
               s_label_select, t_label_select, t_label_select2, []
        # return source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select,\
        #        s_label_select,t_label_select,t_label_select2,torch.cat([target_pchg,target_punchg],dim=0)