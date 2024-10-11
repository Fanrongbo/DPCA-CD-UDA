import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from util.train_util import DIST
from torch.nn.modules.padding import ReplicationPad2d
# from  models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,SynchronizedBatchNorm2d,resnet101_diff
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F


'''self attention block'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query,
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
                 value_out_norm, matmul_norm, with_out_project, **kwargs):
        super(SelfAttentionBlock, self).__init__()
        # key project
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
            use_norm=key_query_norm,
        )
        # query project
        if share_key_query:
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject(
                in_channels=query_in_channels,
                out_channels=transform_channels,
                num_convs=key_query_num_convs,
                use_norm=key_query_norm,
            )
        # value project
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels if with_out_project else out_channels,
            num_convs=value_out_num_convs,
            use_norm=value_out_norm,
        )
        # out project
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                in_channels=transform_channels,
                out_channels=out_channels,
                num_convs=value_out_num_convs,
                use_norm=value_out_norm,
            )
        # downsample
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.transform_channels = transform_channels
    '''forward'''
    def forward(self, query_feats, key_feats):
        #query_feats:torch.Size([2, 512, 32, 32])
        #key_feats:torch.Size([2, 512, 32, 32])
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None: query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context
    '''build project'''
    def buildproject(self, in_channels, out_channels, num_convs, use_norm):
        if use_norm:
            convs = [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    # BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        # BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                    )
                )
        else:
            convs = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels,
                 use_context_within_image=True, num_feats_per_cls=1, use_hard_aggregate=False, memory_data=None,**kwargs):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.use_context_within_image = use_context_within_image
        self.use_hard_aggregate = use_hard_aggregate
        # init memory
        if memory_data is not None:
            self.memory = nn.Parameter(memory_data)
        else:
            self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)#[2, 1, 32]
        # define self_attention module
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self_attention = SelfAttentionBlock(
                    key_in_channels=feats_channels,
                    query_in_channels=feats_channels,
                    transform_channels=transform_channels,
                    out_channels=feats_channels,
                    share_key_query=False,
                    query_downsample=None,
                    key_downsample=None,
                    key_query_num_convs=2,
                    value_out_num_convs=1,
                    key_query_norm=True,
                    value_out_norm=True,
                    matmul_norm=True,
                    with_out_project=True,
                )
                self.self_attentions.append(self_attention)
            self.fuse_memory_conv = nn.Sequential(
                nn.Conv2d(feats_channels * self.num_feats_per_cls, feats_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(feats_channels),
                nn.ReLU(),
            )
        else:
            self.self_attention = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
            )
        # whether need to fuse the contextual information within the input image
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if use_context_within_image:
            self.self_attention_ms = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
            )
            self.bottleneck_ms = nn.Sequential(
                nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
    '''forward'''

    def forward(self, feats, preds=None):
        batch_size, num_channels, h, w = feats.size()
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            # self.memory torch.Size([2, 1, 32])
            memory = self.memory.data[:, idx, :]  # memory [6,512]
            # print('memory',idx,memory.shape)
            selected_memory_list.append(memory.unsqueeze(1))
        # calculate selected_memory according to the num_feats_per_cls
        # false
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                # --(B*H*W, C) --> (B, H, W, C)
                # selected_memory = selected_memory.view(batch_size, h, w, num_channels)#[2, 1, 32]
                # --(B, H, W, C) --> (B, C, H, W)
                # selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
                new_selected_memory = selected_memory
                for b in range(batch_size - 1):
                    new_selected_memory = torch.cat((new_selected_memory, selected_memory), 1)
                selected_memory = new_selected_memory.permute(1, 2, 0).contiguous().unsqueeze(3)
                # print('selected_memory',selected_memory.shape)
                # --append
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
            # --concat
            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            # print('selected_memory',selected_memory.shape)
            selected_memory = self.fuse_memory_conv(selected_memory)#[10, 64, 256, 256])
        else:
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0]
            new_selected_memory = selected_memory
            for b in range(batch_size - 1):
                new_selected_memory = torch.cat((new_selected_memory, selected_memory), 1)
            # --feed into the self attention module
            selected_memory = new_selected_memory.permute(1, 2, 0).contiguous().unsqueeze(3)
            # print('selected_memory', selected_memory.shape)

            selected_memory = self.self_attention(feats, selected_memory)
        # return
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))

        return self.memory.data, memory_output
    '''update'''
    def update(self, mode,features, segmentation, ignore_index=255, strategy='cosine_similarity', learning_rate=None, **kwargs):
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()
        momentum = kwargs['base_momentum']
        if kwargs['adjust_by_learning_rate']:
            momentum = kwargs['base_momentum'] / kwargs['base_lr'] * learning_rate
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()#从小到大排序#label
        # print(mode,'clsids',clsids)
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)

            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]
            # print('feats_cls',feats_cls.shape,self.feats_channels)
            # --init memory by using extracted features
            need_update = True
            for idx in range(self.num_feats_per_cls):
                # print(self.memory[clsid][idx].shape)
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))#这个复制是浅复制，也就是说如果原始张量改变，复制的张量也会改变。
                    need_update = False
                    break
            if not need_update: continue
            # --update according to the selected strategy
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)#updata centroid
                self.memory[clsid].data.copy_(feats_cls)
            else:
                assert strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1),
                    F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = (argmax == idx)
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
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

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels = None):
        super(ASPP, self).__init__()
        if out_channels == None:
            out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res=self.project(res)
        # print('res',res.shape)
        return res
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


    def forward(self, pre_data, post_data):
        #####################
        # decoder
        #####################
        pre_data = pre_data[:, 0:3, :, :]
        post_data = post_data[:, 0:3, :, :]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)
        out=torch.cat([down_feature_41,down_feature_42],1)
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
        # output_feature = self.conv_block_8(concat_feature_8)
        # G-L = F.softmax(output_feature, dim=1)
        # diffout= torch.mean(torch.square(feature_41 - feature_42),dim=1)
        # diffout=torch.mean(torch.abs(feature_41 - feature_42),dim=1)
        # diffout = F.sigmoid(diffout)
        # print('diffout',diffout.shape)

        # output_featurePatch=self.avgpool(G-L)

        return concat_feature_8
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    b, c, h, w = prob.size()
    return -torch.mul(prob, torch.log(prob + 1e-30)) / np.log(c)
class FCSiamDiffDA(nn.Module):
    def __init__(self, in_dim=3,out_dim=2,backbone=None):
        super(FCSiamDiffDA, self).__init__()
        aspp_dilate = [12, 24, 36]#[6, 12, 18]

        self.backbone = FCSiamDiff()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )
        self.bottleneck = nn.Sequential(
            # ASPP(2048, aspp_dilate,out_channels = 512),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.decoder_stage1 = nn.Sequential(
            ASPP(32, aspp_dilate,out_channels=32),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1)
        )
        self.memory_module = FeaturesMemory(
            num_classes=2,
            feats_channels=32,
            transform_channels=32,
            num_feats_per_cls=1,
            out_channels=32,
            use_context_within_image=True,
            use_hard_aggregate=False,
            memory_data=None
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x1,x2, mode, target, lr, i_iter):
        img_size = x1.size(2), x1.size(3)
        input_shape = x1.shape[-2:]
        #feature:torch.Size([4, 2048, 32, 32])
        features = self.backbone(x1,x2)#[10, 32, 256, 256]
        # feed to memory
        preds_stage1 = self.decoder_stage1(features)#[10, 2, 256, 256])
        preds_stage2 = None
        tored_memory = None
        #迭代次数大于等于3000次的时候，开始更新memory，保证此时的特征是不变特征。
        if i_iter > 3000 or mode == 'TEST':
            # memory_output:torch.Size([4, 256, 32, 32])
            memory_input = self.bottleneck(features)
            tored_memory, memory_output = self.memory_module(memory_input, preds_stage1)
            # x:torch.Size([4, 6, 32, 32])
            preds_stage2 = self.classifier(memory_output)
            preds_stage2 = F.interpolate(preds_stage2, size=input_shape, mode='bilinear', align_corners=False)
            if mode == 'TRAIN':
                # updata memory
                with torch.no_grad():
                    self.memory_module.update(
                        mode=mode,
                        features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                        segmentation=target,
                        learning_rate=lr,
                        strategy='cosine_similarity',
                        ignore_index=255,
                        base_momentum=0.9,
                        base_lr=0.01,
                        adjust_by_learning_rate=True,
                    )
            if mode == 'TARGET':
                # updata memory
                target_tar = preds_stage2.detach().max(dim=1)[1]
                entropy = prob_2_entropy(F.softmax(preds_stage2.detach(),dim=1))
                entropy = torch.sum(entropy, axis=1)  # 2,512,512
                # # # #
                # # #高斯爬升曲线参数
                # t = i_iter * 10e-5
                # arfa = (1 - math.exp(-0.05 * t)) / (1 + math.exp(-0.05 * t))
                arfa = 0.3
                target_tar[entropy > arfa] = 255
                with torch.no_grad():
                    self.memory_module.update(
                        mode=mode,
                        features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                        segmentation=target_tar,
                        learning_rate=lr,
                        strategy='cosine_similarity',
                        ignore_index=255,
                        base_momentum=0.9,
                        base_lr=0.01,
                        adjust_by_learning_rate=True,
                    )
        preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=False)
        return tored_memory, preds_stage1, preds_stage2