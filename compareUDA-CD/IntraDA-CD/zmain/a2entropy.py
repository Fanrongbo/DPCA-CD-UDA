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
def convert_colormap(label, colormap, num_clc):
    height = label.shape[0]
    width = label.shape[1]
    label_resize = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    label_resize = label_resize
    img_color = np.zeros([height, width, 3], dtype=np.uint8)

    for idx in range(num_clc):
        img_color[label_resize == idx] = colormap[idx]

    return img_color
def colorize_save(output_pt_tensor, colormap, num_clc, name):
    output_np_tensor = output_pt_tensor.cpu().data[0].numpy()
    mask_np_tensor = output_np_tensor.transpose(1, 2, 0)
    mask_np_tensor = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    # print('output_np_tensor',mask_np_tensor.shape)
    # print(np.sum(mask_np_tensor))

    mask_np_tensor = mask_np_tensor
    mask_np_color = convert_colormap(mask_np_tensor, colormap, num_clc)
    mask_Img = Image.fromarray(mask_np_tensor)
    mask_color = Image.fromarray(mask_np_color)
    name = name.split('/')[-1]
    # if np.sum(mask_np_tensor)>500:
    mask_Img.save('./G-L/color_masks/%s' % (name))
    mask_color.save('./G-L/color_masks/%s_color.png' % (name.split('.')[0]))
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
def cluster_subdomain(entropy_list, lambda1):
    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]

    easy_split = entropy_rank[: int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank) * lambda1):]

    with open('G-L/easy_split.txt', 'w+') as f:
        for item in easy_split:
            # print('item',item)
            f.write('%s\n' % item)

    with open('G-L/hard_split.txt', 'w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)

    return copy_list
def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    cmap[0, 0] = 0
    cmap[0, 1] = 0
    cmap[0, 2] = 0

    cmap[1, 0] = 255
    cmap[1, 1] = 255
    cmap[1, 2] = 255



    return cmap
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

    saveLast=True
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
    saveroot = '/data/project_frb/DA/DACDCompare/CDDA/CCDA_LGFAAdventCD/zmain/log/FCSiamDiff/20230917-17_19_GZ_CDPatch/'#
    save_path = saveroot + '/savemodel/_9_acc-0.8766_chgAcc-0.4534_unchgAcc-0.9354.pth'#
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

    t_loaderDict = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
            opt.t = i
            opt.phase = 'target'
            t_loader = CreateDataLoader(opt)
            t_loaderDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            t_data = t_loader.load_data()
            t_data_len = len(t_data)
            t_size = len(t_loader)
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size))
    print(t_loaderDict)

    tool = CDModelutil(opt)
    # cd_model = create_model(opt)
    # initialize model
    model_state_dict = None
    fx_pretrained = True
    resume_dict = None
    ######################################################################################################
    if opt.load_pretrain:
        # saveroot='./log/CD_DA_building/HLCDNet2_a/20230316-23_34_GZ_CDPatch'
        # save_path=saveroot+'/savemodel/_101_acc-0.9808_chgAcc-0.9316_unchgAcc-0.9849.pth'
        # figure_train_metrics = load_pickle(saveroot+"/fig_train.pkl")
        # figure_val_metrics = load_pickle(saveroot+"/fig_val.pkl")
        # start_epoch = len(figure_train_metrics['nochange_acc'])+1
        # print('start_epochstart_epochstart_epoch',start_epoch,'end:',opt.num_epochs + opt.num_decay_epochs + 1)
        model_state_dict,bn_domain_map,optimizer_state_dict=tool.load_ckpt(save_path)
        # opt.num_epochs=opt.num_epochs+start_epoch
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
    t_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[opt.t]]
    t_data = t_loader.load_data()
    t_data_len = len(t_data)
    tbar = tqdm(range(t_data_len))
    iter_t = iter(t_data)
    opt.phase = 'val'
    from advent.utils.func import prob_2_entropy

    colormap = labelcolormap(2)
    entropy_list = []

    net.eval()
    ce_lossT = 0
    focalLossT = 0
    diceLossT = 0
    LossT = 0
    for i in tbar:
        with torch.no_grad():
        # for i, data in enumerate(val_data):
        #     data=iter_val.next()
            data_test = next(iter_t)
            data_T1_val = Variable(data_test['t1_img']).to(DEVICE)
            data_T2_val = Variable(data_test['t2_img']).to(DEVICE)

            name=data_test['t1_path']
            # print('name',name)
            cd_val_pred = net.forward(data_T1_val, data_T2_val)
            pred_trg_entropy = prob_2_entropy(F.softmax(cd_val_pred[0]))
            label = Variable(data_test['label']).to(DEVICE)
            entropy_list.append((name[0], pred_trg_entropy.mean().item()))
            colorize_save(cd_val_pred[0], colormap, 2, name[0])
            # if i>10:
            #    break
    cluster_subdomain(entropy_list, 0.67)





