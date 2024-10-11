import os
import time
import random
import numpy as np
from data.data_loader import CreateDataLoader
# from model.cd_model import *
from util.train_util import *

from option.train_options import TrainOptions
from option.base_options import BaseOptions
from util.visualizer import Visualizer
from util.metric_tool import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool import get_parser_with_args,initialize_weights,save_pickle,load_pickle
from util.drawTool import setFigure,add_weight_decay,plotFigureCD,MakeRecordFloder,confuseMatrix,plotFigure
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils
from PIL import Image
import cv2
import heapq

class Class_Features:
    def __init__(self, numbers=19):
        self.class_numbers = numbers
        self.tsne_data = 0
        self.pca_data = 0
        # self.class_features = np.zeros((19, 256))
        self.centroids = np.zeros((10, 2, 32)).astype('float32')
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.all_vectors = []
        self.pred_ids = []
        self.ids = []
        self.pred_num = np.zeros(numbers + 1)
        self.labels = ["unchg","chg"]
        self.markers = ["*", "o"]
        return
    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 2, w, h).cuda()
        id = torch.where(label < 1, label, torch.Tensor([1]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1
    def calculate_mean_vector(self, feat_cls, outputs, labels_val, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        # outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax= torch.argmax(outputs_softmax.detach(), dim=1,keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        # outputs_pred = model.process_pred(outputs_softmax, 0.5)
        # outputs_pred = outputs_argmax[:, 0:19, :, :] * outputs_softmax
        labels_expanded = self.process_label(labels_val)
        outputs_pred = labels_expanded * outputs_argmax
        # print(labels_val.shape, labels_expanded.shape, outputs_pred.shape)
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean(self, ):
        out = [[] for i in range(self.class_numbers)]
        for i in range(self.class_numbers):
            out[i] = self.class_features[i] / max(self.num[i], 1)
        return out

    def calculate_dis(self, vector, id):
        if isinstance(vector, torch.Tensor): vector = vector.detach().cpu().numpy().squeeze()
        mean = self.calculate_mean()
        dis = []
        for i in range(self.class_numbers):
            dis_vec = [x - y for x, y in zip(mean[i], vector)]
            dis.append(np.linalg.norm(dis_vec, 2))
        return dis
    def calculate_min_mse(self, single_image_objective_vectors):
        loss = []
        for centroid in self.centroids:
            new_loss = np.mean((single_image_objective_vectors - centroid) ** 2)
            loss.append(new_loss)
        min_loss = min(loss)
        # min_index = loss.index(min_loss)
        # print(min_loss)
        # print(min_index)
        return min_loss
if __name__ == '__main__':
    gpu_id="0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False,gpu=gpu_id)
    opt.num_epochs=3
    opt.batch_size=1
    opt.use_ce_loss=True
    opt.use_hybrid_loss=False
    opt.num_decay_epochs=0
    opt.dset='CD_DA_building'
    opt.model_type='FCSiamDiff'
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest=True

    name = opt.dset + 'CDBase/' + opt.model_type + ''
    opt.LChannel=True
    opt.dataroot=opt.dataroot+'/'+opt.dset
    opt.s = 0
    opt.t=1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'CD_DA_building':
        # cfg.TRAINLOG.DATA_NAMES = ['SYSU_CD', 'LEVIR_CDPatch', 'GZ_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch', 'SYSU_CD', 'GZ_CDPatch']
        cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatch']
    # saveroot=None//
    saveroot = '/data/project_frb/DA/DACDCompare/EasytoHardAdvent/zmain/log/FCSiamDiff/20230917-17_19_GZ_CDPatch/'
    save_path = saveroot + '/savemodel/_9_acc-0.8766_chgAcc-0.4534_unchgAcc-0.9354.pth'
    opt.load_pretrain = True

    print('\n########## Recording File Initialization#################')
    SEED = opt.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg.TRAINLOG.STARTTIME = time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime("%Y%m%d-%H_%M_"+cfg.TRAINLOG.DATA_NAMES[opt.s], time.localtime())
    filename_main = os.path.basename(__file__)

    print('\n########## Load the Target Dataset #################')

    opt.phase = 'train'
    train_loader = CreateDataLoader(opt)
    train_data = train_loader.load_data()
    train_data_len = len(train_data)
    train_size = len(train_loader)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], train_size))

    tool = CDModelutil(opt)
    # cd_model = create_model(opt)
    # initialize model
    model_state_dict = None
    fx_pretrained = True
    resume_dict = None
    ######################################################################################################
    if opt.load_pretrain:
        model_state_dict,bn_domain_map,optimizer_state_dict=tool.load_ckpt(save_path)
        cfg.DA.BN_DOMAIN_MAP = bn_domain_map
    else:
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}

    print('\n########## Build the Molde #################')
    opt.phase = 'train'
    net = cfg.TRAINLOG.NETWORK_DICT[opt.model_type]()
    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(DEVICE)
    print('DEVICE:', DEVICE)
    if model_state_dict is not None:
        model_utils.init_weights(net, model_state_dict, bn_domain_map, False)
    ########################################################
    print('target:', cfg.TRAINLOG.DATA_NAMES[opt.t])
    t_loader = train_loader
    t_data = t_loader.load_data()
    t_data_len = len(t_data)
    tbar = tqdm(range(t_data_len))
    iter_t = iter(t_data)
    opt.phase = 'val'
    entropy_list = []

    net.eval()
    ce_lossT = 0
    focalLossT = 0
    diceLossT = 0
    LossT = 0
    class_features = Class_Features(numbers=2)
    full_dataset_objective_vectors = np.zeros([t_data_len, 2, 32])
    CAU_full = torch.load('./output/G-L/anchors/Target_cluster_centroids_full_10.pkl')
    CAU_full = CAU_full.reshape(10, 2, 32)
    class_features.centroids = CAU_full
    cac_list = []
    # sample=[]
    namelist=[]
    index=[]
    for i in tbar:
        with torch.no_grad():
            # for i, data in enumerate(val_data):
            #     data=iter_val.next()
            data_test = next(iter_t)
            data_T1_val = Variable(data_test['t1_img']).to(DEVICE)
            data_T2_val = Variable(data_test['t2_img']).to(DEVICE)
            label = Variable(data_test['label']).to(DEVICE)

            name = data_test['t1_path']
            cd_predT, defeat3T, defeatTDA = net.forward(data_T1_val, data_T2_val)

            vectors, ids = class_features.calculate_mean_vector(defeatTDA, cd_predT, label, net)
            single_image_objective_vectors = np.zeros([2, 32])
            for t in range(len(ids)):
                single_image_objective_vectors[ids[t]] = vectors[t].detach().cpu().numpy().squeeze()
                # model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')
            MSE = class_features.calculate_min_mse(single_image_objective_vectors)
            # print(single_image_objective_vectors)
            full_dataset_objective_vectors[i, :] = single_image_objective_vectors[:]
            # print('single_image_objective_vectors[:]',single_image_objective_vectors[:].shape)
            # sample.append({'cac_list': MSE, 'name': name, 'index': i})

            cac_list.append(MSE)
            namelist.append(name)
            index.append(i)

            # if i>10:
            #    break
    # torch.save(full_dataset_objective_vectors, './output/G-L/Source_objective_vectors.pkl')
    lenth = len(cac_list)
    per = 0.05
    selected_lenth = int(per * lenth)
    print('selected_lenth',selected_lenth)
    selected_index_list = list(map(cac_list.index, heapq.nlargest(selected_lenth, cac_list)))
    # print(selected_index_list)

    selected_index_list.sort()
    selected_img_list = []
    for index in selected_index_list:
        selected_img_list.append(namelist[index])
    file = open(os.path.join('./output/G-L', 'stage1_cac_list_%.2f.txt' % per), 'w')
    for i in range(len(selected_img_list)):
        # print(selected_img_list[i])
        file.write(selected_img_list[i][0] + '\n')


