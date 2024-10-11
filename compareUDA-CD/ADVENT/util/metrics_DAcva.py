import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.Weight import Weight
class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def CORAL(self,source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        # print('source',source.shape)#18, 1024]
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss

    def selecdata(self,feature, label):
        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((1, 0, 2, 3)), start_dim=1, end_dim=3)
        label_index = torch.nonzero(label_flatten)

        label_index = torch.flatten(label_index)
        label_index_rand = torch.randperm(label_index.nelement())
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[:, label_index[0]].unsqueeze(0)
        return feature_flatten_select, label_index, feature_flatten
    def forward(self, source, target,label_source,pred_target):
        chgthreshold=800
        unchgthreshold=800
        H, W = source.size(2), source.size(3)
        label_source = F.interpolate(label_source.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        pred_target = F.interpolate(pred_target.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        ones = torch.ones_like(label_source)
        zeros = torch.zeros_like(label_source)
        label_source = torch.where(label_source > 0.5, ones, zeros)
        pred_target = torch.where(pred_target > 0.5, ones, zeros)
        ############### change origin
        # source = (label_source.repeat([1, source.shape[1], 1, 1])).float()
        # target = (pred_target.repeat([1, target.shape[1], 1, 1])).float()
        source_chg_flatten_select,source_chg_index,source_chg_flatten=self.selecdata(source,label_source)
        target_chg_flatten_select,target_chg_index,target_chg_flatten=self.selecdata(target,pred_target)
        # one=torch.ones_like(source_chg_flatten[:,1])

        # print('source_chg_flatten_select',source_chg_flatten_select.shape)
        if source_chg_index.shape[0]<chgthreshold or target_chg_index.shape[0]<chgthreshold:
            chgthreshold= np.minimum(source_chg_index.shape[0],target_chg_index.shape[0])
            # print('chgthreshold',chgthreshold)
        source_chg_flatten_select=source_chg_flatten[:,source_chg_index[0:chgthreshold]]
        target_chg_flatten_select=target_chg_flatten[:, target_chg_index[0:chgthreshold]]


        ###############################################
        ######################unchange
        # source = ((1-label_source).repeat([1, source.shape[1], 1, 1])).float()
        # target = ((1-pred_target).repeat([1, target.shape[1], 1, 1])).float()
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1-label_source)
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1-pred_target)
        # one = torch.ones_like(source_unchg_flatten[:, 1])

        # print('source_unchg_flatten_select', source_unchg_flatten_select.shape)
        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        source_unchg_flatten_select=source_unchg_flatten[:, source_unchg_index[0:unchgthreshold]]
        target_unchg_flatten_select=target_unchg_flatten[:, target_unchg_index[0:unchgthreshold]]
        CORAL_value_chg = self.CORAL(source_chg_flatten_select, target_chg_flatten_select)
        CORAL_value_unchg = self.CORAL(source_unchg_flatten_select, target_unchg_flatten_select)

        return CORAL_value_chg + CORAL_value_unchg

def CORAL_ori(source, target):
    source = torch.flatten(source, start_dim=1, end_dim=3)[0:2,:]
    target = torch.flatten(target, start_dim=1, end_dim=3)[0:2,:]

    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=2):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        # print(f_of_X.shape,f_of_Y.shape)
        # f_of_X=torch.flatten(f_of_X,start_dim=1,end_dim=3)
        # f_of_Y=torch.flatten(f_of_Y,start_dim=1,end_dim=3)

        delta = (f_of_X.float().mean(0) - f_of_Y.float().mean(0))
        # print(delta.shape)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):

        if self.kernel_type == 'linear':
            linear_mmd2_value=self.linear_mmd2(source, target)
            return  linear_mmd2_value
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class MMD_lossclass(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=2):
        super(MMD_lossclass, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y,label_source,pred_target):
        loss = 0.0
        # print(f_of_X.shape,f_of_Y.shape)
        label_source_1_num = torch.sum(label_source, [2, 3])
        pred_target_1_num = torch.sum(pred_target, [2, 3])
        f_of_X=torch.flatten(f_of_X,start_dim=1,end_dim=3)
        f_of_Y=torch.flatten(f_of_Y,start_dim=1,end_dim=3)
        # print('pred_target_1_num',pred_target_1_num.shape,pred_target_1_num)
        # delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        delta = f_of_X.float()/label_source_1_num - f_of_Y.float()/pred_target_1_num
        print('f_of_X',f_of_X.shape,pred_target_1_num.shape,delta.shape)

        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target,label_source,pred_target):
        H, W = source.size(2), source.size(3)
        # total=torch.tensor(source.shape[2]*source.shape[3])
        # print(pred_target)
        print(source.shape, target.shape,label_source.shape,pred_target.shape)
        label_source = F.interpolate(label_source.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        pred_target = F.interpolate(pred_target.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        print(source.shape, target.shape,label_source.shape,pred_target.shape)

        # label_source_chg_num = torch.sum(label_source, [1, 2])
        # pred_target_chg_num = torch.sum(pred_target, [1, 2])
        # label_source_unchg_num=total-label_source_chg_num
        # pred_target_unchg_num=total-pred_target_chg_num
        # pixelNum=[total,label_source_chg_num,pred_target_chg_num]
        if self.kernel_type == 'linear':
            linear_mmd2_value_chg=self.linear_mmd2(label_source*source, pred_target * target, label_source,pred_target)

            linear_mmd2_value_unchg=self.linear_mmd2((1-label_source)*source, (1-pred_target) * target, (1-label_source),(1-pred_target))

            return  linear_mmd2_value_chg+linear_mmd2_value_unchg
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
class MMD_lossclass2(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=2):
        super(MMD_lossclass2, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        # loss = 0.0
        # print(f_of_X.shape,f_of_Y.shape)
        # label_source_1_num = torch.sum(label_source, [2, 3])
        # pred_target_1_num = torch.sum(pred_target, [2, 3])

        # f_of_X=torch.flatten(f_of_X,start_dim=1,end_dim=3)
        # f_of_Y=torch.flatten(f_of_Y,start_dim=1,end_dim=3)
        # print('f_of_X',f_of_X.shape,f_of_Y.shape)
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        # delta = f_of_X.float()/label_source_1_num - f_of_Y.float()/pred_target_1_num
        # print('f_of_X',f_of_X.shape,pred_target_1_num.shape,delta.shape)

        loss = delta.dot(delta.T)
        return loss
    def selecdata(self,feature,label):
        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((1, 0, 2, 3)), start_dim=1, end_dim=3)
        label_index = torch.nonzero(label_flatten)

        label_index = torch.flatten(label_index)
        label_index_rand = torch.randperm(label_index.nelement())
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[:, label_index[0]].unsqueeze(0)
        return  feature_flatten_select,label_index,feature_flatten

    def forward(self, source, target,label_source,pred_target):
        chgthreshold=1000
        unchgthreshold=1000
        H, W = source.size(2), source.size(3)
        label_source = F.interpolate(label_source.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        pred_target = F.interpolate(pred_target.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        ones = torch.ones_like(label_source)
        zeros = torch.zeros_like(label_source)
        label_source = torch.where(label_source > 0.5, ones, zeros)
        pred_target = torch.where(pred_target > 0.5, ones, zeros)
        ############### change origin
        # print('source',source.shape)
        # source = (label_source.repeat([1, source.shape[1], 1, 1])*2).float()
        # print('source',source.shape)
        # target = (pred_target.repeat([1, target.shape[1], 1, 1])).float()
        source_chg_flatten_select,source_chg_index,source_chg_flatten=self.selecdata(source,label_source)
        target_chg_flatten_select,target_chg_index,target_chg_flatten=self.selecdata(target,pred_target)
        # one=torch.ones_like(source_chg_flatten[:,1])

        # print('source_chg_flatten_select',source_chg_flatten_select.shape)
        if source_chg_index.shape[0]<chgthreshold or target_chg_index.shape[0]<chgthreshold:
            chgthreshold= np.minimum(source_chg_index.shape[0],target_chg_index.shape[0])
            # print('chgthreshold',chgthreshold)
        source_chg_flatten_select=source_chg_flatten[:,source_chg_index[0:chgthreshold]]
        target_chg_flatten_select=target_chg_flatten[:, target_chg_index[0:chgthreshold]]


        ###############################################
        ######################unchange
        # source = ((1-label_source).repeat([1, source.shape[1], 1, 1])* 2).float()
        # target = ((1-pred_target).repeat([1, target.shape[1], 1, 1])).float()
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1-label_source)
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1-pred_target)
        # one = torch.ones_like(source_unchg_flatten[:, 1])

        # print('source_unchg_flatten_select', source_unchg_flatten_select.shape)
        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        source_unchg_flatten_select=source_unchg_flatten[:, source_unchg_index[0:unchgthreshold]]
        target_unchg_flatten_select=target_unchg_flatten[:, target_unchg_index[0:unchgthreshold]]

        if self.kernel_type == 'linear':
            #
            # print('source_chg_flatten',source_chg_flatten_select.shape, f_of_Y.shape)

            linear_mmd2_value_chg = self.linear_mmd2(source_chg_flatten_select, target_chg_flatten_select)
            linear_mmd2_value_unchg=self.linear_mmd2(source_unchg_flatten_select,target_unchg_flatten_select)

            return  linear_mmd2_value_chg+linear_mmd2_value_unchg
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class MMD_lossclass3(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_lossclass3, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
#高斯核函数是两个向量欧式距离的单调函数。σ 是带宽，控制径向作用范围，换句话说，σ 控制高斯核函数的局部作用范围。
    # 当 x 和 x′ 的欧式距离处于某一个区间范围内的时候，假设固定 x′，k(x,x′) 随 x 的变化而变化的相当显著。
#高斯核函数的核心思想是将每一个样本点映射到一个无穷维的特征空间，从而使得原本线性不可分的数据线性能够线性可分。
    def L2dis(self,source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))  # [1,64,256]->[64, 64, 256]
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))  # [64,1,256]->[64, 64, 256]

        L2_distance = ((total0 - total1) ** 2)  # calculate the distance of source and target data;
        L2_distance=L2_distance.sum(2)

        return L2_distance

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))#[1,64,256]->[64, 64, 256]
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))#[64,1,256]->[64, 64, 256]

        L2_distance = ((total0-total1)**2).sum(2)#calculate the distance of source and target data;
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        # print('bandwidth_list',bandwidth_list)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        # kernel_val = [L2_distance / bandwidth_temp
        #               for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def selecdata(self, feature, label):
        #label 6, 1, 128, 128
        #feature 6, 32, 128, 128

        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)
        label_index = torch.nonzero(label_flatten)
        label_index = torch.flatten(label_index)
        label_index_rand = torch.randperm(label_index.nelement())
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[label_index,:]#bs,c

        return feature_flatten_select, label_index, feature_flatten

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss
#边缘
    def marginal(self, source, target):

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
#条件
    def conditionalnew(self, source, target, s_label, t_label,DEVICE, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        # print('source',source.shape, target.shape, s_label.shape, t_label.shape)
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = Weight.cal_weight(
            s_label, t_label, type='visual')
        weight_ss = torch.from_numpy(weight_ss).to(DEVICE)
        weight_tt = torch.from_numpy(weight_tt).to(DEVICE)
        weight_st = torch.from_numpy(weight_st).to(DEVICE)

        ##
        kernels = self.guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = torch.Tensor([0]).to(DEVICE)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        _, t_label_l = torch.max(t_label, 1)
        # print('t_label_l',t_label_l.shape,t_label_l)
        # print('s_label',s_label.shape,s_label)

        label = torch.cat([s_label, t_label_l])
        label_ex0 = label.unsqueeze(0).expand(
            int(label.size(0)), int(label.size(0)))  # [64,1,256]->[64, 64, 256]
        label_ex1 = label.unsqueeze(1).expand(
            int(label.size(0)), int(label.size(0)))  # [64,1,256]->[64, 64, 256]
        eqOut = label_ex0.eq(label_ex1)
        eqOut = eqOut.int()
        one = torch.ones_like(eqOut)
        weight = torch.where(eqOut == 1, one, -one)
        weight_oness = weight[:batch_size, :batch_size]
        weight_oness = weight_oness - torch.diag_embed(torch.diag(weight_oness))
        weight_onett = weight[batch_size:, batch_size:]
        weight_onett = weight_onett - torch.diag_embed(torch.diag(weight_onett))
        weight_onest = weight[:batch_size, batch_size:]
        # print('weight_onest* weight_st * ST',(weight_onest* weight_st * ST).sum())
        # print('weight_onett * weight_tt * TT',(weight_onett * weight_tt * TT).sum())
        # print( 'weight_oness* weight_ss * SS',(weight_oness* weight_ss * SS).sum())

        loss = torch.mean( -2 *weight_onest* weight_st * ST + weight_onett * weight_tt * TT + weight_oness* weight_ss * SS)
        return loss
    def conditional(self, source, target, s_label, t_label,DEVICE, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        # print('source',source.shape, target.shape, s_label.shape, t_label.shape)
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = Weight.cal_weight(
            s_label, t_label, type='visual')
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        source_max, _ = torch.max(source, dim=0, keepdim=True)
        target_max, _ = torch.max(target, dim=0, keepdim=True)
        # # source_max = torch.mean(source, dim=1, keepdim=True)
        # # target_max = torch.mean(target, dim=1, keepdim=True)
        # #
        source = source / (source_max + 0.001)
        target = target / (target_max + 0.001)
        # print((torch.cat([source,target]).mean(0)).shape)
        var = torch.var(torch.cat([source,target],0).mean(0), dim=0)
        # print(var.shape)
        # var=0
        # source_unchg=source*s_label
        # print('s_label',s_label.shape)
        loss_unchg=(1-s_label.unsqueeze(1))*(torch.exp(source)-1)
        loss_unchg=loss_unchg.sum()/(1-s_label).sum()

        loss_chg=s_label.unsqueeze(1)*(torch.exp(-source))
        loss_chg=loss_chg.sum()/s_label.sum()

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        _, t_label_l = torch.max(t_label, 1)
        # print('t_label_l',t_label_l.shape,t_label_l)
        # print('s_label',s_label.shape,s_label)

        label = torch.cat([s_label, t_label_l])
        label_ex0 = label.unsqueeze(0).expand(
            int(label.size(0)), int(label.size(0)))  # [64,1,256]->[64, 64, 256]
        label_ex1 = label.unsqueeze(1).expand(
            int(label.size(0)), int(label.size(0)))  # [64,1,256]->[64, 64, 256]
        eqOut = label_ex0.eq(label_ex1)
        eqOut = eqOut.int()
        one = torch.ones_like(eqOut)
        weight = torch.where(eqOut == 1, one, -one)
        weight_oness = weight[:batch_size, :batch_size]
        weight_oness = weight_oness - torch.diag_embed(torch.diag(weight_oness))
        weight_onett = weight[batch_size:, batch_size:]
        weight_onett = weight_onett - torch.diag_embed(torch.diag(weight_onett))
        weight_onest = weight[:batch_size, batch_size:]
        # print('weight_onett',weight_onett)
        # loss = torch.sum(-2 * weight_onest * weight_st * ST)
        # loss = torch.sum(-2 * weight_onest * weight_st * ST)
        # loss = torch.sum( -4 *weight_onest* weight_st * ST - weight_onett * weight_tt * TT - weight_oness* weight_ss * SS)+var
        # loss = torch.sum( -4 *weight_onest* weight_st * ST- weight_onett * weight_tt * TT - weight_oness* weight_ss * SS)+var+loss_unchg+loss_chg
        loss = torch.sum( -4 *weight_onest* weight_st * ST- weight_onett * weight_tt * TT - weight_oness* weight_ss * SS)

        #positive or negative
        # _, t_label_l = torch.max(t_label, 1)
        # label = torch.cat([s_label, t_label_l])
        # label_ex0 = label.unsqueeze(0).expand(
        #     int(label.size(0)), int(label.size(0)))  # [64,1,256]->[64, 64, 256]
        # label_ex1 = label.unsqueeze(1).expand(
        #     int(label.size(0)), int(label.size(0)))  # [64,1,256]->[64, 64, 256]
        # eqOut = label_ex0.eq(label_ex1)
        #
        # eqOut = eqOut.int()
        # one = torch.ones_like(eqOut)
        # weight = torch.where(eqOut == 1, one, -one)
        # weight_oness = weight[:batch_size, :batch_size]
        # weight_onett = weight[batch_size:, batch_size:]
        # weight_onest = weight[:batch_size, batch_size:]
        #
        # print('weight_onest',weight_onest)
        # print('weight_onett', weight_onett)
        # print('weight_oness', weight_oness)
        # loss_st = torch.sum(weight_st * torch.exp(weight_onest * ST))
        # loss_ss = torch.sum(weight_ss * torch.exp(weight_oness * SS))
        # loss_tt = torch.sum(weight_tt * torch.exp(weight_onett * TT))
        # # loss=loss_st
        # loss = 2*loss_st+loss_ss+loss_tt



        # one=torch.ones((batch_size//2,batch_size//2))
        # ST_1=torch.cat([-1*one,one],dim=1)
        # ST_2=torch.cat([one,-1*one],dim=1)
        # ST_=torch.cat([ST_1,ST_2],dim=0).to(DEVICE)
        # _, t_label_l = torch.max(t_label, 1)
        #
        # s_label_ex = s_label.unsqueeze(0).expand(
        #     int(s_label.size(0)), int(s_label.size(0)))  # [64,1,256]->[64, 64, 256]
        # t_label_l_ex = t_label_l.unsqueeze(1).expand(
        #     int(s_label.size(0)), int(s_label.size(0)))  # [1,64,256]->[64, 64, 256]
        # eqOut = t_label_l_ex.eq(s_label_ex)
        # eqOut = eqOut.int()
        # one = torch.ones_like(eqOut)
        # weight = torch.where(eqOut == 1, one, -one)
        #
        # loss_st = torch.sum(weight_st * torch.exp(weight * ST))
        # loss_ss = torch.sum(torch.exp(weight_ss * SS * weight))
        # loss_tt = torch.sum(torch.exp(weight_tt * TT * weight))
        #
        # loss += loss_st
        # print('ST_',batch_size,ST_)
        # ST_loss=torch.sum(torch.exp(-2 * weight_st *ST*ST_)-1)
        # TT_loss=torch.sum(torch.exp(-1*weight_tt *TT*ST_)-1)
        # SS_loss=torch.sum(torch.exp(-1*weight_ss *SS*ST_)-1)
        # loss += ST_loss+TT_loss+SS_loss

        return loss,var
    def conditional2(self, source_chg,source_unchg,target_chg,target_unchg, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        # print('source_chg',source_chg.shape, target_chg.shape, source_unchg.shape, target_unchg.shape)
        batch_size_chg = source_chg.size()[0]
        num_total=(2*batch_size_chg)*(2*batch_size_chg)
        if source_chg.shape[0]==0:
            # print('source_chg',source_chg,target_chg)
            loss_chg = torch.Tensor([0]).cuda()
        else:
            kernels_chg = self.guassian_kernel(source_chg, target_chg,
                                               kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            kernels_chg_uchg = self.guassian_kernel(source_chg, target_unchg,
                                               kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            if torch.sum(torch.isnan(sum(kernels_chg))):
                loss_chg = torch.Tensor([0]).cuda()
            else:
                # batch_size_chg = source_chg.size()[0]
                # SS_chg = kernels_chg[:batch_size_chg, :batch_size_chg]/(batch_size_chg*batch_size_chg)
                # TT_chg = kernels_chg[batch_size_chg:, batch_size_chg:]/(batch_size_chg*batch_size_chg)
                # ST_chg = kernels_chg[:batch_size_chg, batch_size_chg:]/(batch_size_chg*batch_size_chg)
                loss_chg = torch.sum(kernels_chg)/num_total
            if torch.sum(torch.isnan(sum(kernels_chg_uchg))):
                loss_chg_uchg = torch.Tensor([0]).cuda()
            else:
                loss_chg_uchg = torch.sum(kernels_chg_uchg)/num_total


        if source_unchg.shape[0]==0:
            # print('source_unchg',source_unchg,target_unchg)
            loss_unchg = torch.Tensor([0]).cuda()
        else:
            kernels_unchg = self.guassian_kernel(source_unchg, target_unchg,
                                                 kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            kernels_unchg_chg = self.guassian_kernel(source_unchg, target_chg,
                                                 kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            if torch.sum(torch.isnan(sum(kernels_unchg))):
                loss_unchg = torch.Tensor([0]).cuda()
            else:
                # batch_size_unchg = source_unchg.size()[0]
                # SS_unchg = kernels_unchg[:batch_size_unchg, :batch_size_unchg]/(batch_size_unchg*batch_size_unchg)
                # TT_unchg = kernels_unchg[batch_size_unchg:, batch_size_unchg:]/(batch_size_unchg*batch_size_unchg)
                # ST_unchg = kernels_unchg[:batch_size_unchg, batch_size_unchg:]/(batch_size_unchg*batch_size_unchg)
                loss_unchg = torch.sum(kernels_unchg)/num_total
            if torch.sum(torch.isnan(sum(kernels_unchg_chg))):
                loss_uchg_chg = torch.Tensor([0]).cuda()
            else:
                loss_uchg_chg = torch.sum(kernels_unchg_chg)/num_total


        return loss_chg+loss_unchg-loss_uchg_chg-loss_chg_uchg
    def conditional3(self, source_chg,source_unchg,target_chg,target_unchg, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        # print('source_chg',source_chg.shape, target_chg.shape, source_unchg.shape, target_unchg.shape)
        source = torch.cat([source_chg.mean(1), source_unchg.mean(1)], 0)#2400
        target = torch.cat([target_chg.mean(1), target_unchg.mean(1)], 0)
        # print('target',target.shape)
        var_source = torch.var(source)#
        var_target = torch.var(target)
        # print('var_source', var_source, var_target)
        loss_var = torch.exp(-var_source) + torch.exp(-var_target)


        batch_size_chg = source_chg.size()[0]
        # print(source_chg.shape)
        source_chg_max,_=torch.max(source_chg,dim=1,keepdim=True)
        source_unchg_max,_=torch.max(source_unchg,dim=1,keepdim=True)
        target_chg_max,_=torch.max(target_chg,dim=1,keepdim=True)
        target_unchg_max,_=torch.max(target_unchg,dim=1,keepdim=True)

        source_chg=source_chg/(source_chg_max+0.001)
        source_unchg=source_unchg/(source_unchg_max+0.001)
        target_chg=target_chg/(target_chg_max+0.001)
        target_unchg=target_unchg/(target_unchg_max+0.001)


        # print(source_chg.shape)
        num_total=(batch_size_chg)*(batch_size_chg)
        if source_chg.shape[0]==0:
            # print('source_chg',source_chg,target_chg)
            loss_chg = torch.Tensor([0]).cuda()
            loss_chg_uchg = torch.Tensor([0]).cuda()
        else:
            # kernels_chg = self.guassian_kernel(source_chg, target_chg,
            #                                   kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            # kernels_chg_uchg = self.guassian_kernel(source_chg, target_unchg,
            #                                         kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            kernels_chg=self.L2dis(source_chg, target_chg)
            kernels_chg_uchg=self.L2dis(source_chg, target_unchg)
            if torch.sum(torch.isnan(sum(kernels_chg))):
                loss_chg = torch.Tensor([0]).cuda()
            else:
                loss_chg = torch.mean(torch.exp(kernels_chg)-1)
            if torch.sum(torch.isnan(sum(kernels_chg_uchg))):
                loss_chg_uchg = torch.Tensor([0]).cuda()
            else:
                # ST_chg_uchg = kernels_chg_uchg[:batch_size_chg, batch_size_chg:]
                # loss_chg_uchg = torch.sum(kernels_chg_uchg)/num_total
                loss_chg_uchg = torch.mean(torch.exp(-kernels_chg_uchg))
        if source_unchg.shape[0]==0:
            loss_unchg = torch.Tensor([0]).cuda()
            loss_uchg_chg = torch.Tensor([0]).cuda()
        else:
            # kernels_chg = self.guassian_kernel(source_unchg, target_unchg,
            #                                   kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            # kernels_chg_uchg = self.guassian_kernel(source_unchg, target_chg,
            #                                         kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

            kernels_unchg = self.L2dis(source_unchg, target_unchg)
            kernels_unchg_chg = self.L2dis(source_unchg, target_chg)
            if torch.sum(torch.isnan(sum(kernels_unchg))):
                loss_unchg = torch.Tensor([0]).cuda()
            else:
                loss_unchg = torch.mean(torch.exp(kernels_unchg) - 1)
            if torch.sum(torch.isnan(sum(kernels_unchg_chg))):
                loss_uchg_chg = torch.Tensor([0]).cuda()
            else:
                loss_uchg_chg = torch.mean(torch.exp(-kernels_unchg_chg))

        loss=loss_chg+10*loss_unchg+loss_uchg_chg+loss_chg_uchg+loss_var

        return loss,var_source,var_target
        # return  loss_unchg - loss_uchg_chg

    def conditional4(self, source_chg,source_unchg,target_chg,target_unchg, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        # print('source_chg',source_chg.shape, target_chg.shape, source_unchg.shape, target_unchg.shape)

        # source_feature_select = torch.cat((source_chg_flatten_select, source_unchg_flatten_select), dim=0)
        # source_chg = source_feature_select[0:1]
        source = torch.cat([source_chg.mean(1), source_unchg.mean(1)], 0)#2400
        target = torch.cat([target_chg.mean(1), target_unchg.mean(1)], 0)
        # print('target',target.shape)
        var_source = torch.var(source)#
        var_target = torch.var(target)
        # print('var_source', var_source, var_target)
        loss_var = torch.exp(-var_source) + torch.exp(-var_target)
        # target_unchg=target_unchg*t_label_select[:,0]
        # target_chg = target_chg * t_label_select[:, 1]

        # batch_size_chg = source_chg.size()[0]
        # print(source_chg.shape)
        source_chg_max,_=torch.max(source_chg,dim=1,keepdim=True)
        source_unchg_max,_=torch.max(source_unchg,dim=1,keepdim=True)
        target_chg_max,_=torch.max(target_chg,dim=1,keepdim=True)
        target_unchg_max,_=torch.max(target_unchg,dim=1,keepdim=True)
        source_chg=source_chg/(source_chg_max+0.001)
        source_unchg=source_unchg/(source_unchg_max+0.001)
        target_chg=target_chg/(target_chg_max+0.001)
        target_unchg=target_unchg/(target_unchg_max+0.001)


        # print(source_chg.shape)
        # num_total=(batch_size_chg)*(batch_size_chg)
        if source_chg.shape[0]==0:
            # print('source_chg',source_chg,target_chg)
            loss_chg = torch.Tensor([0]).cuda()
            loss_chg_uchg = torch.Tensor([0]).cuda()
        else:

            kernels_chg=self.L2dis(source_chg, target_chg)
            kernels_chg_uchg=self.L2dis(source_chg, target_unchg)
            if torch.sum(torch.isnan(sum(kernels_chg))):
                loss_chg = torch.Tensor([0]).cuda()
            else:
                loss_chg = torch.mean(torch.exp(kernels_chg)-1)
            if torch.sum(torch.isnan(sum(kernels_chg_uchg))):
                loss_chg_uchg = torch.Tensor([0]).cuda()
            else:
                # ST_chg_uchg = kernels_chg_uchg[:batch_size_chg, batch_size_chg:]
                # loss_chg_uchg = torch.sum(kernels_chg_uchg)/num_total
                loss_chg_uchg = torch.mean(torch.exp(-kernels_chg_uchg))


        if source_unchg.shape[0]==0:
            loss_unchg = torch.Tensor([0]).cuda()
            loss_uchg_chg = torch.Tensor([0]).cuda()
        else:
            kernels_unchg = self.L2dis(source_unchg, target_unchg)
            kernels_unchg_chg = self.L2dis(source_unchg, target_chg)
            if torch.sum(torch.isnan(sum(kernels_unchg))):
                loss_unchg = torch.Tensor([0]).cuda()
            else:
                loss_unchg = torch.mean(torch.exp(kernels_unchg) - 1)
            if torch.sum(torch.isnan(sum(kernels_unchg_chg))):
                loss_uchg_chg = torch.Tensor([0]).cuda()
            else:
                loss_uchg_chg = torch.mean(torch.exp(-kernels_unchg_chg))

        loss=loss_chg+10*loss_unchg+loss_uchg_chg+loss_chg_uchg+loss_var

        return loss,var_source,var_target
        # return  loss_unchg - loss_uchg_chg

    def select_feature(self,source, target, s_label,pred_target_label, t_label_out):
        chgthreshold = 800#select 1000 pixel
        unchgthreshold = 800
        chgthreshold_t = 800  # select 1000 pixel
        unchgthreshold_t = 800
        H, W = source.size(2), source.size(3)

        label_source = F.interpolate(s_label.unsqueeze(1).float(), size=(H, W), mode='bilinear',
                                     align_corners=False)
        pred_target = F.interpolate(pred_target_label.unsqueeze(1).float(), size=(H, W), mode='bilinear',
                                    align_corners=False)

        ones = torch.ones_like(label_source)
        zeros = torch.zeros_like(label_source)
        label_source = torch.where(label_source > 0.5, ones, zeros)
        pred_target = torch.where(pred_target > 0.5, ones, zeros)
        s_label = label_source
        ######################change feature
        # source = (label_source.repeat([1, source.shape[1], 1, 1])*2).float()
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, label_source)
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pred_target)
        # print(source_chg_flatten_select[0], source_chg_index[0], source_chg_flatten[0])

        if source_chg_index.shape[0] < chgthreshold or target_chg_index.shape[0] < chgthreshold_t:
            chgthreshold = np.minimum(source_chg_index.shape[0], target_chg_index.shape[0])
            chgthreshold_t=chgthreshold
        source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold_t],:]#bs,c
        # print('source_chg_flatten_select',source_chg_flatten_select.shape)
        ''' verity
        # print('source_chg_flatten_select',source_chg_flatten_select.shape,target_chg_flatten_select.shape)
        one = torch.ones_like(source_chg_flatten[ 1,:])*2
        for i in range(1,source_chg_index.shape[0]):
            # target_chg_flatten_select = torch.cat((target_chg_flatten_select, target_chg_flatten[:, target_chg_index[i]].unsqueeze(0)), 0)
            if not (source_chg_flatten[ source_chg_index[i],:].equal(one)):
                print('b',source_chg_flatten[ source_chg_index[i],:].shape, source_chg_flatten[ source_chg_index[i],:])
            # else:
            #     print('c', source_chg_flatten[source_chg_index[i], :].shape, source_chg_flatten[source_chg_index[i], :])
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
         '''

        #############change label
        label_source_flatten = torch.flatten(s_label, start_dim=0, end_dim=3)
        t_label = F.interpolate(t_label_out.float(), size=(H, W), mode='bilinear',
                                align_corners=False).permute((0, 2, 3, 1))#[bs,2,h,w]->[bs,h,w,2]
        t_label = torch.flatten(t_label, start_dim=0, end_dim=2)#bs,2
        s_label_chg_select = label_source_flatten[source_chg_index[0:chgthreshold]]#[bs]
        t_label_chg_select = t_label[target_chg_index[0:chgthreshold_t]]#[bs,2]
        # print('t_label_chg_select',t_label_chg_select)
        '''verify
        pred_target_flatten = torch.flatten(pred_target, start_dim=0, end_dim=3)
        pred_target_chg_select = pred_target_flatten[target_chg_index[0:chgthreshold]]  # [bs,2]
        _, pred_target = torch.max(t_label_chg_select, 1)
        # if result most equal zeros,the method is right!
        # if you want the result of whole zeros, the original size of t_label,s_label and pred_target should be input!
        print(pred_target-pred_target_chg_select)
        '''
        ###############################################
        ######################unchange
        # source = ((1 - label_source).repeat([1, source.shape[1], 1, 1])).float()
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - label_source)
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1 - pred_target)
        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < chgthreshold_t:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
            unchgthreshold_t=unchgthreshold
        source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold],:]#bs,c
        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold_t],:]#bs,c
        '''verity
        print('source_unchg_flatten_select', source_unchg_flatten_select.shape, target_unchg_flatten_select.shape)
        one = torch.ones_like(source_unchg_flatten[1,:])
        for i in range(1, source_unchg_index.shape[0]):
            if not (source_unchg_flatten[source_unchg_index[i],:].equal(one)):
                print('c', source_unchg_flatten[source_unchg_index[i],:].shape, source_unchg_flatten[source_unchg_index[i],:])
        '''
        #############unchange label


        s_label_unchg_select = label_source_flatten[source_unchg_index[0:unchgthreshold]]  # [bs]
        t_label_unchg_select = t_label[target_unchg_index[0:unchgthreshold_t]]
        # print('t_label_unchg_select',t_label_unchg_select)
        '''verify
        pred_target_flatten = torch.flatten(pred_target, start_dim=0, end_dim=3)
        pred_target_chg_select = pred_target_flatten[target_unchg_index[0:unchgthreshold]]  # [bs,2]
        _, pred_target = torch.max(t_label_unchg_select, 1)
        # if result most equal zeros,the method is right!
        # if you want the result of whole zeros, the original size of t_label,s_label and pred_target should be input!
        print(pred_target_chg_select)
        print(pred_target-pred_target_chg_select)
                '''
        source_feature_select=torch.cat((source_chg_flatten_select,source_unchg_flatten_select),dim=0)
        target_feature_select=torch.cat((target_chg_flatten_select,target_unchg_flatten_select),dim=0)#[bs*2,c]
        s_label_select=torch.cat((s_label_chg_select,s_label_unchg_select),dim=0)
        t_label_select = torch.cat((t_label_chg_select, t_label_unchg_select), dim=0)
        # print(s_label_select,t_label_select)
        # print('source_feature_selec22222t',source_feature_select.shape)
        return source_feature_select,target_feature_select,s_label_select,t_label_select
    def select_feature_unban(self,source, target, s_label,pred_target_label, t_label_out):
        chgthreshold = 800#select 1000 pixel
        unchgthreshold = 800
        chgthreshold_t = 800  # select 1000 pixel
        unchgthreshold_t = 800
        H, W = source.size(2), source.size(3)

        label_source = F.interpolate(s_label.unsqueeze(1).float(), size=(H, W), mode='bilinear',
                                     align_corners=False)
        pred_target = F.interpolate(pred_target_label.unsqueeze(1).float(), size=(H, W), mode='bilinear',
                                    align_corners=False)

        ones = torch.ones_like(label_source)
        zeros = torch.zeros_like(label_source)
        label_source = torch.where(label_source > 0.5, ones, zeros)
        pred_target = torch.where(pred_target > 0.5, ones, zeros)
        s_label = label_source
        ######################change feature
        # source = (label_source.repeat([1, source.shape[1], 1, 1])*2).float()
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, label_source)
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pred_target)
        # print(source_chg_flatten_select[0], source_chg_index[0], source_chg_flatten[0])

        if source_chg_index.shape[0] < chgthreshold or target_chg_index.shape[0] < chgthreshold_t:
            chgthreshold = np.minimum(source_chg_index.shape[0], target_chg_index.shape[0])
            chgthreshold_t=chgthreshold
        source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold_t],:]#bs,c
        # print('source_chg_flatten_select',source_chg_flatten_select.shape)
        ''' verity
        # print('source_chg_flatten_select',source_chg_flatten_select.shape,target_chg_flatten_select.shape)
        one = torch.ones_like(source_chg_flatten[ 1,:])*2
        for i in range(1,source_chg_index.shape[0]):
            # target_chg_flatten_select = torch.cat((target_chg_flatten_select, target_chg_flatten[:, target_chg_index[i]].unsqueeze(0)), 0)
            if not (source_chg_flatten[ source_chg_index[i],:].equal(one)):
                print('b',source_chg_flatten[ source_chg_index[i],:].shape, source_chg_flatten[ source_chg_index[i],:])
            # else:
            #     print('c', source_chg_flatten[source_chg_index[i], :].shape, source_chg_flatten[source_chg_index[i], :])
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
         '''

        #############change label
        label_source_flatten = torch.flatten(s_label, start_dim=0, end_dim=3)
        t_label = F.interpolate(t_label_out.float(), size=(H, W), mode='bilinear',
                                align_corners=False).permute((0, 2, 3, 1))#[bs,2,h,w]->[bs,h,w,2]
        t_label = torch.flatten(t_label, start_dim=0, end_dim=2)#bs,2
        s_label_chg_select = label_source_flatten[source_chg_index[0:chgthreshold]]#[bs]
        t_label_chg_select = t_label[target_chg_index[0:chgthreshold_t]]#[bs,2]
        # print('t_label_chg_select',t_label_chg_select)
        '''verify
        pred_target_flatten = torch.flatten(pred_target, start_dim=0, end_dim=3)
        pred_target_chg_select = pred_target_flatten[target_chg_index[0:chgthreshold]]  # [bs,2]
        _, pred_target = torch.max(t_label_chg_select, 1)
        # if result most equal zeros,the method is right!
        # if you want the result of whole zeros, the original size of t_label,s_label and pred_target should be input!
        print(pred_target-pred_target_chg_select)
        '''
        ###############################################
        ######################unchange
        # source = ((1 - label_source).repeat([1, source.shape[1], 1, 1])).float()
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - label_source)
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1 - pred_target)
        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < chgthreshold_t:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
            unchgthreshold_t=unchgthreshold
        source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold],:]#bs,c
        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold_t],:]#bs,c
        '''verity
        print('source_unchg_flatten_select', source_unchg_flatten_select.shape, target_unchg_flatten_select.shape)
        one = torch.ones_like(source_unchg_flatten[1,:])
        for i in range(1, source_unchg_index.shape[0]):
            if not (source_unchg_flatten[source_unchg_index[i],:].equal(one)):
                print('c', source_unchg_flatten[source_unchg_index[i],:].shape, source_unchg_flatten[source_unchg_index[i],:])
        '''
        #############unchange label


        s_label_unchg_select = label_source_flatten[source_unchg_index[0:unchgthreshold]]  # [bs]
        t_label_unchg_select = t_label[target_unchg_index[0:unchgthreshold_t]]

        # if unchgthreshold!=unchgthreshold_t:
        #     print(unchgthreshold, unchgthreshold_t)
        #     threshold = np.minimum(unchgthreshold, unchgthreshold_t)
        #
        #     s_label_unchg_select=s_label_unchg_select[0:threshold]
        #     t_label_unchg_select = t_label_unchg_select[0:threshold]
        #     source_chg_flatten_select = source_chg_flatten_select[0:threshold]
        #     source_unchg_flatten_select = source_unchg_flatten_select[0:threshold]
        #     target_chg_flatten_select = target_chg_flatten_select[0:threshold]
        #     target_unchg_flatten_select = target_unchg_flatten_select[0:threshold]
        # print('t_label_unchg_select',t_label_unchg_select)
        '''verify
        pred_target_flatten = torch.flatten(pred_target, start_dim=0, end_dim=3)
        pred_target_chg_select = pred_target_flatten[target_unchg_index[0:unchgthreshold]]  # [bs,2]
        _, pred_target = torch.max(t_label_unchg_select, 1)
        # if result most equal zeros,the method is right!
        # if you want the result of whole zeros, the original size of t_label,s_label and pred_target should be input!
        print(pred_target_chg_select)
        print(pred_target-pred_target_chg_select)
                '''
        source_feature_select=torch.cat((source_chg_flatten_select,source_unchg_flatten_select),dim=0)
        target_feature_select=torch.cat((target_chg_flatten_select,target_unchg_flatten_select),dim=0)#[bs*2,c]
        s_label_select=torch.cat((s_label_chg_select,s_label_unchg_select),dim=0)
        t_label_select = torch.cat((t_label_chg_select, t_label_unchg_select), dim=0)
        # print(s_label_select,t_label_select)
        # print('source_feature_selec22222t',source_feature_select.shape)
        return source_feature_select,target_feature_select,s_label_select,t_label_select
        # conditional_=True
        # if conditional_:
        #     loss_chg=self.conditional(source_feature_select,target_feature_select, s_label_select, t_label_select)
        #     print('loss_chg',loss_chg)
    def select_feature2(self,source, target, s_label,pred_target_label, t_label_out):
        chgthreshold = 1200#select 1000 pixel
        unchgthreshold = 1200
        H, W = source.size(2), source.size(3)

        s_label = s_label.unsqueeze(1).float()
        pred_target_label = pred_target_label.unsqueeze(1).float()
        # print('pred_target_label',pred_target_label)
        if s_label.size(2) != source.size(2):
            label_source = F.interpolate(s_label, size=(H, W), mode='bilinear',
                                         align_corners=False)
            pred_target = F.interpolate(pred_target_label, size=(H, W), mode='bilinear',
                                        align_corners=False)

            ones = torch.ones_like(label_source)
            zeros = torch.zeros_like(label_source)
            label_source = torch.where(label_source > 0.5, ones, zeros)
            pred_target = torch.where(pred_target > 0.5, ones, zeros)
            # s_label = label_source
        else:
            pred_target=pred_target_label
            label_source=s_label
        ######################change feature
        # source = (label_source.repeat([1, source.shape[1], 1, 1])*2).float()
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, label_source)
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pred_target)

        if source_chg_index.shape[0] < chgthreshold or target_chg_index.shape[0] < chgthreshold:
            chgthreshold = np.minimum(source_chg_index.shape[0], target_chg_index.shape[0])
        source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold],:]#bs,c
        # print('source_chg_flatten_select',source_chg_flatten_select.shape)
        ''' verity
        # print('source_chg_flatten_select',source_chg_flatten_select.shape,target_chg_flatten_select.shape)
        one = torch.ones_like(source_chg_flatten[ 1,:])*2
        for i in range(1,source_chg_index.shape[0]):
            # target_chg_flatten_select = torch.cat((target_chg_flatten_select, target_chg_flatten[:, target_chg_index[i]].unsqueeze(0)), 0)
            if not (source_chg_flatten[ source_chg_index[i],:].equal(one)):
                print('b',source_chg_flatten[ source_chg_index[i],:].shape, source_chg_flatten[ source_chg_index[i],:])
            # else:
            #     print('c', source_chg_flatten[source_chg_index[i], :].shape, source_chg_flatten[source_chg_index[i], :])
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
         '''

        #############change label
        # label_source_flatten = torch.flatten(s_label, start_dim=0, end_dim=3)
        t_label = F.interpolate(t_label_out.float(), size=(H, W), mode='bilinear',
                                align_corners=False).permute((0, 2, 3, 1))#[bs,2,h,w]->[bs,h,w,2]
        t_label = torch.flatten(t_label, start_dim=0, end_dim=2)#bs,2

        # s_label_chg_select = label_source_flatten[source_chg_index[0:chgthreshold]]#[bs]
        t_label_chg_select = t_label[target_chg_index[0:chgthreshold]]#[bs,2]
        # print('t_label_chg_select',t_label_chg_select)
        '''verify
        pred_target_flatten = torch.flatten(pred_target, start_dim=0, end_dim=3)
        pred_target_chg_select = pred_target_flatten[target_chg_index[0:chgthreshold]]  # [bs,2]
        _, pred_target = torch.max(t_label_chg_select, 1)
        # if result most equal zeros,the method is right!
        # if you want the result of whole zeros, the original size of t_label,s_label and pred_target should be input!
        print(pred_target-pred_target_chg_select)
        '''
        ###############################################
        ######################unchange
        # source = ((1 - label_source).repeat([1, source.shape[1], 1, 1])).float()
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - label_source)
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1 - pred_target)
        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold],:]#bs,c
        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold],:]#bs,c
        '''verity
        print('source_unchg_flatten_select', source_unchg_flatten_select.shape, target_unchg_flatten_select.shape)
        one = torch.ones_like(source_unchg_flatten[1,:])
        for i in range(1, source_unchg_index.shape[0]):
            if not (source_unchg_flatten[source_unchg_index[i],:].equal(one)):
                print('c', source_unchg_flatten[source_unchg_index[i],:].shape, source_unchg_flatten[source_unchg_index[i],:])
        '''
        #############unchange label


        # s_label_unchg_select = label_source_flatten[source_unchg_index[0:unchgthreshold]]  # [bs]
        t_label_unchg_select = t_label[target_unchg_index[0:unchgthreshold]]
        # print('t_label_unchg_select',t_label_unchg_select)
        '''verify
        pred_target_flatten = torch.flatten(pred_target, start_dim=0, end_dim=3)
        pred_target_chg_select = pred_target_flatten[target_unchg_index[0:unchgthreshold]]  # [bs,2]
        _, pred_target = torch.max(t_label_unchg_select, 1)
        # if result most equal zeros,the method is right!
        # if you want the result of whole zeros, the original size of t_label,s_label and pred_target should be input!
        print(pred_target_chg_select)
        print(pred_target-pred_target_chg_select)
                '''
        # print('t_label_unchg_select', t_label_unchg_select.shape,target_chg_flatten_select.shape,t_label_unchg_select[:,0].shape)
        target_unchg_flatten_select=target_unchg_flatten_select*t_label_unchg_select[:,0].unsqueeze(1)
        target_chg_flatten_select = target_chg_flatten_select * t_label_chg_select[:,1].unsqueeze(1)

        # source_feature_select=torch.cat((source_chg_flatten_select,source_unchg_flatten_select),dim=0)
        # target_feature_select=torch.cat((target_chg_flatten_select,target_unchg_flatten_select),dim=0)#[bs*2,c]
        # s_label_select=torch.cat((s_label_chg_select,s_label_unchg_select),dim=0)
        # t_label_select = torch.cat((t_label_chg_select, t_label_unchg_select), dim=0)
        # print(s_label_select,t_label_select)
        # print('source_feature_selec22222t',source_feature_select.shape)
        return source_chg_flatten_select,source_unchg_flatten_select,target_chg_flatten_select,target_unchg_flatten_select
class CVACD(nn.Module):
    def __init__(self):
        super(CVACD, self).__init__()

    def cva(self, T1, T2):
        norm=True
        if norm:
            stdT1 = torch.std(T1, dim=[2, 3], keepdim=True)+0.0001
            stdT2 = torch.std(T2, dim=[2, 3], keepdim=True)+0.0001
            meanT1 = torch.mean(T1, dim=[2, 3], keepdim=True)
            meanT2 = torch.mean(T2, dim=[2, 3], keepdim=True)
            # print(T1.shape,meanT1.shape,stdT1.shape)
            normT1 = (T1 - meanT1) / stdT1
            normT2 = (T2 - meanT2) / stdT2
            # normT1=self.norm(T1)
            # imgT1=torch.var()
            img_diff = torch.square(normT1 - normT2)
            L2_norm = torch.sqrt(img_diff)
        else:
            img_diff = torch.square(T1 - T2)
            L2_norm = torch.sqrt(img_diff)
        # print(stdT1.shape,L2_norm.shape)
        return L2_norm
    def otsu(self,mask_in):
        # normalize
        mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
                    (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])

        bs = mask_in.shape[0]
        h = mask_in.shape[1]
        mask = []
        maskP=[]
        for i in range(bs):
            threshold_t = 0.
            max_g = 0.
            for t in range(5):
                mask_i = mask_norm[i]
                low = mask_i[mask_i < t * 0.1]
                high = mask_i[mask_i >= t * 0.1]
                low_num = low.shape[0] / h
                high_num = high.shape[0] / h
                low_mean = low.mean()
                high_mean = high.mean()

                g = low_num * high_num * ((low_mean - high_mean) ** 2)
                if g > max_g:
                    max_g = g
                    threshold_t = t * 0.1
            # print('ma',mask_i.shape)
            maskp=mask_i.clone()

            mask_i[mask_i < threshold_t] = 0
            mask_i[mask_i > threshold_t] = 1
            # maskP_chg=maskp*mask_i
            maskP_chg=(0.5+(maskp-threshold_t)/(1-threshold_t)*0.5)*mask_i
            maskp_unchg=(0.5+(1-maskp/threshold_t)*0.5)*(1-mask_i)
            maskP_chgo=maskP_chg+(1-maskp_unchg)
            maskp_unchgo=maskp_unchg+(1-maskP_chg)
            # print(threshold_t,maskp_unchg.max(),maskp_unchg.min(),maskP_chg.max(),maskP_chg.min())
            maskP.append(torch.cat([maskp_unchgo, maskP_chgo], dim=0))
            # maskP.append(torch.cat([maskP_chgo, maskp_unchgo], dim=0))

            mask.append(mask_i)
            # print(maskp)
        mask_out = torch.stack(mask, dim=0)
        mask_outP = torch.stack(maskP, dim=0)
        # mask_outP=torch.cat([1-mask_outP,mask_outP],dim=1)
        # print(mask_outP)
        return mask_out,mask_outP

    def forward(self,TT1,TT2,size):
        # cvaOut = self.cva(TT1[:, 3, :, :], TT2[:, 3, :, :])
        cvaOut = self.cva(TT1, TT2)
        cvaOut=torch.sum(cvaOut,dim=1).unsqueeze(1)
        mask_out,mask_outP = self.otsu(cvaOut)
        mask_out=abs(nn.functional.interpolate(mask_out,(size,size)))
        mask_outP = nn.functional.interpolate(mask_outP, (size, size))
        return mask_out,mask_outP

class SelecFeat():
    def selecdata(self, feature, label):
        #label 6, 1, 128, 128
        #feature 6, 32, 128, 128

        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)
        label_index = torch.nonzero(label_flatten)
        label_index = torch.flatten(label_index)
        label_index_rand = torch.randperm(label_index.nelement())
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[label_index,:]#bs,c

        return feature_flatten_select, label_index, feature_flatten
    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def select_featureST(self,source,s_label,target,pseudo_label,softmaxLabel,softLog,p=0,pe=0,device='cuda'):
        # source=source.reshape(source.shape[0], source.shape[1], -1)
        # s_label = s_label.reshape(s_label.shape[0], -1)
        # target=target.reshape(target.shape[0], target.shape[1], -1)

        # print('pseudo_label',pseudo_label.shape,softmaxLabel.shape)
        # [softLogS, softLogT]=softLog
        self.device=device
        chgthreshold = 1200 # select 1000 pixel
        unchgthreshold = 1200
        self.chgthreshold=chgthreshold
        self.unchgthreshold=unchgthreshold
        pseudo_label=pseudo_label.unsqueeze(1)
        # print('softmaxLabel',softmaxLabel.shape)#[13, 2, 65536]
        # softmaxLabelori=softmaxLabel.reshape(-1,2,s_label.shape[2],s_label.shape[3])#[bs,2,h,w]->[bs,h,w,2]
        softmaxLabelori=softmaxLabel
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
        ones=torch.ones_like(pseudo_label)
        zeros=torch.zeros_like(pseudo_label)

        pseudo_labeltChg=torch.where(softmaxLabelori[:,1,:,:].unsqueeze(1)>p,pseudo_label,zeros)############################3############################3############################3############################3############################3
        # pseudo_labeltChg=torch.where((softmaxLabelori[:,1,:,:]/softmaxLabelori[:,1,:,:].max()).unsqueeze(1)>p,pseudo_label,zeros)
        # print('pseudo_label',pseudo_labeltChg.shape,pseudo_label.shape,pseudo_label.sum(),pseudo_labeltChg.sum())
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pseudo_labeltChg)


        if source_chg_index.shape[0] < chgthreshold or target_chg_index.shape[0] < chgthreshold:
            chgthreshold = np.minimum(source_chg_index.shape[0], target_chg_index.shape[0])
        source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold],:]#bs,c
        softmaxLabel_chg_select = softmaxLabel[target_chg_index[0:chgthreshold]]  # [bs,2]

        # softLogT_chg_select = softLogT[target_chg_index[0:chgthreshold]]  # [bs,2]
        # softLogS_chg_select = softLogS[source_chg_index[0:chgthreshold]]
        # print(softmaxLabel_chg_select)
        # print('softmaxLabel_chg_select',softmaxLabel_chg_select.shape)
        # target_chg_flatten_selectW = target_chg_flatten_select * softmaxLabel_chg_select[:, 1].unsqueeze(1)
####################unchg
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - s_label)
        # print('softmaxLabel',softmaxLabel.shape)
        pseudo_labeltunChg = torch.where(softmaxLabelori[:,0,:,:].unsqueeze(1)>p, pseudo_label, ones)############################3############################3############################3############################3
        # pseudo_labeltunChg = torch.where((softmaxLabelori[:,0,:,:]/softmaxLabelori[:,0,:,:].max()).unsqueeze(1)>p, pseudo_label, ones)


        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1 - pseudo_labeltunChg)


        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        if unchgthreshold > chgthreshold:
            unchgthreshold = chgthreshold
        source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold], :]  # bs,c
        # softLogS_unchg_select=softLogS[source_unchg_index[0:unchgthreshold]]

        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold], :]  # bs,c
        softmaxLabel_unchg_select = softmaxLabel[target_unchg_index[0:unchgthreshold]]
        # softLogT_unchg_select = softLogT[target_unchg_index[0:unchgthreshold]]

        # target_unchg_flatten_selectW = target_unchg_flatten_select * softmaxLabel_unchg_select[:, 0].unsqueeze(1)#weight
        self.chgNum = chgthreshold
        self.unchgNum = unchgthreshold
        unchglabel = self.to_onehot(torch.zeros_like(softmaxLabel_unchg_select[:, 0]).long(), 2)
        chglabel = self.to_onehot(torch.ones_like(softmaxLabel_unchg_select[:, 1]).long(), 2)
        # print(unchglabel, chglabel)
        # print('s',softmaxLabel_unchg_select.shape,softmaxLabel_chg_select[1].shape,softmaxLabel_unchg_select[:,0].min(),softmaxLabel_chg_select[:,1].min())
        s_label_select = torch.cat([unchglabel,chglabel], dim=0)
        # print('s_label_select',s_label_select)
        t_label_select = torch.cat([softmaxLabel_unchg_select,softmaxLabel_chg_select-pe ], dim=0)
        # print('softmaxLabel_unchg_select',t_label_select.shape)
        t_label_select2=torch.cat([softmaxLabel_unchg_select,softmaxLabel_chg_select ], dim=0)
        # softLogS=torch.cat([softLogS_unchg_select,softLogS_chg_select], dim=0)
        # softLogT=torch.cat([softLogT_unchg_select,softLogT_chg_select], dim=0)

        # print('t_label_select2',t_label_select2.shape)
        return source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select,s_label_select,t_label_select,t_label_select2,[0,0]
    def cva(self,T1,T2):
        stdT1 = torch.std(T1, dim=[1, 2],keepdim=True)
        stdT2 = torch.std(T2, dim=[1, 2],keepdim=True)
        meanT1=torch.mean(T1,dim=[1,2],keepdim=True)
        meanT2=torch.mean(T2,dim=[1,2],keepdim=True)
        # print(T1.shape,meanT1.shape,stdT1.shape)
        normT1=(T1-meanT1)/stdT1
        normT2 = (T2 - meanT2) / stdT2
        # normT1=self.norm(T1)
        # imgT1=torch.var()
        img_diff=torch.square(normT1-normT2)
        L2_norm=torch.sqrt(img_diff)
        # print(stdT1.shape,L2_norm.shape)
        return L2_norm

    def otsu(self,mask_in):
        # normalize
        mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
                    (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])

        bs = mask_in.shape[0]
        h = mask_in.shape[1]
        mask = []
        for i in range(bs):
            threshold_t = 0.
            max_g = 0.
            for t in range(10):
                mask_i = mask_norm[i]
                low = mask_i[mask_i < t * 0.1]
                high = mask_i[mask_i >= t * 0.1]
                low_num = low.shape[0] / h
                high_num = high.shape[0] / h
                low_mean = low.mean()
                high_mean = high.mean()

                g = low_num * high_num * ((low_mean - high_mean) ** 2)
                if g > max_g:
                    max_g = g
                    threshold_t = t * 0.1

            mask_i[mask_i < threshold_t] = 0
            mask_i[mask_i > threshold_t] = 1
            mask.append(mask_i)
        mask_out = torch.stack(mask, dim=0)

        return mask_out

