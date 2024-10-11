import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
from data.data_loader import CreateDataLoader
from modelDA.retinex import retinex_synthesis
from util.train_util import *
from option.train_options import TrainOptions
from util.visualizer import Visualizer
from util.metric_tool2 import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool import get_parser_with_args, initialize_weights, save_pickle, load_pickle
from util.drawTool import setFigureval, plotFigureTarget, plotFigureDA, MakeRecordFloder, confuseMatrix, plotFigure, \
    setFigureDA
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils
from modelDA.discriminator import FCDiscriminator,FCDiscriminatorLow,FCDiscriminatorHigh,FCDiscriminatorLowMask,FCDiscriminatorHighMask
from util.func import prob_2_entropy
from model.Siamese_diff import FCSiamDiffMaskAttention,FCSiamDiffMaskAttentionCenter,pseudoMultiCenter
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE


def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, num):
    LEARNING_RATE = 2.5e-4
    lr = lr_poly(LEARNING_RATE, i_iter, num, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter, num):
    lr = lr_poly(1e-4, i_iter, num, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self):
        '''
        features: (m,n)
        labels: (m,)
        '''


    def plot_tsne(self, features,labels, save_eps=True):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        self.features = features
        self.labels = labels
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        print('tsne',labels.shape,features.shape)
        X_pca = PCA(n_components=3)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        # plt.clim(-0.5, 3.5)
        # cbar = plt.colorbar(ticks=range(2))
        print('data',data.shape)
        del features
        for i in range(data.shape[0]):
            if label[i]==0:
                plt.scatter(data[i, 0], data[i, 1], c='green', alpha=0.6,
                        cmap=plt.cm.get_cmap('rainbow', 2))
            elif label[i] == 1:
                plt.scatter(data[i, 0], data[i, 1], c='blue', alpha=0.6,
                            cmap=plt.cm.get_cmap('rainbow', 2))
            elif label[i] == 2:
                plt.scatter(data[i, 0], data[i, 1], c='red', alpha=0.6,
                            cmap=plt.cm.get_cmap('rainbow', 2))
                plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                         color=plt.cm.Set1(self.labels[i] / 10.),
                         fontdict={'weight': 'bold', 'size': 9})
            elif label[i] == 3:
                plt.scatter(data[i, 0], data[i, 1], c='yellow', alpha=0.6,
                            cmap=plt.cm.get_cmap('rainbow', 2))
                plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                         color=plt.cm.Set1(self.labels[i] / 10.),
                         fontdict={'weight': 'bold', 'size': 9})
            elif label[i] == 4:
                plt.scatter(data[i, 0], data[i, 1], c='black', alpha=0.6,
                            cmap=plt.cm.get_cmap('rainbow', 2))
                plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                         color=plt.cm.Set1(self.labels[i] / 10.),
                         fontdict={'weight': 'bold', 'size': 9})
            elif label[i] == 5:
                plt.scatter(data[i, 0], data[i, 1], c='magenta', alpha=0.6,
                            cmap=plt.cm.get_cmap('rainbow', 2))
                plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                         color=plt.cm.Set1(self.labels[i] / 10.),
                         fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('G-Ltsne.tif', dpi=600, format='eps')
        plt.show()

if __name__ == '__main__':

    # time.sleep(3000)
    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False, gpu=gpu_id)
    opt.num_epochs = 30
    opt.batch_size = 14
    opt.use_ce_loss = True
    opt.use_hybrid_loss = False
    opt.num_decay_epochs = 0
    opt.dset = 'CD_DA_building'
    opt.model_type = '2-3CenterAtten'
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest = False
    saveLast = True
    name = opt.dset + 'CDPretrain2/' + opt.model_type + 'EntropyMidGAN-itermultiCenter-New'

    opt.LChannel = False
    opt.dataroot = opt.dataroot + '/' + opt.dset
    opt.s = 0
    opt.t = 1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'CD_DA_building':
        # cfg.TRAINLOG.DATA_NAMES = ['SYSU_CD', 'LEVIR_CDPatch', 'GZ_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['2LookingPatch256', 'LEVIR_CDPatch', 'WHbuilding256']
        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['S2LookingPatchNoUnchg', 'WHBuildingNoUnchg']

        # cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch', 'GZ_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatchNoResolution']

        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatchNoResolution', 'LEVIR_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'WHBuildingNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuilding3', 'GDbuilding']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'S2LookingPatchNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['GDbuilding', 'WHBuildingNoUnchg']

        cfg.TRAINLOG.DATA_NAMES = [ 'WHBuildingNoUnchg','GDbuilding']
    Nepoch = 10

    opt.load_pretrain = True
    if opt.load_pretrain:
        #
        # saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-itermultiCenter-New/20231123-17_38_G-LE-paste-iterTh02/'
        # save_path = saveroot + '/savemodel/_21_acc-0.9231_chgAcc-0.6302_unchgAcc-0.9630.pth'
        # saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-itermultiCenter-New/20231123-17_37_L-GZ-paste-iterTh02/'
        # save_path = saveroot + '/savemodel/_21_acc-0.9780_chgAcc-0.8006_unchgAcc-0.9858.pth'
        # saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-itermultiCenter-New/20231125-12_21_W-GD-pasteMean-iterTh015Adjust/'
        # save_path = saveroot + '/savemodel/_31_acc-0.9805_chgAcc-0.8733_unchgAcc-0.9849.pth'
        saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-itermultiCenter-New/20231124-23_49_G-WH-pasteMean-iterTh02/'
        save_path = saveroot + '/savemodel/_31_acc-0.9888_chgAcc-0.8701_unchgAcc-0.9939.pth'
        opt.num_epochs = 20
    else:
        saveroot = None
        save_path = None

    cfg.TRAINLOG.EXCEL_LOGSheet = ['wsTrain', 'wsVal']
    for wsN in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if wsN == opt.s:
            continue
        else:
            cfg.TRAINLOG.EXCEL_LOGSheet.append('wsT-' + cfg.TRAINLOG.DATA_NAMES[wsN][0])
            # wsT = [cfg.TRAINLOG.EXCEL_LOG['wsT1'], cfg.TRAINLOG.EXCEL_LOG['wsT2']]
    print('\n########## Recording File Initialization#################')
    SEED = 1240#opt.SEED
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg.TRAINLOG.STARTTIME = time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime(
        "%Y%m%d-%H_%M_" + cfg.TRAINLOG.DATA_NAMES[opt.s][0] + '-' + cfg.TRAINLOG.DATA_NAMES[opt.t][:2] + '-Test',
        time.localtime())
    filename_main = os.path.basename(__file__)
    start_epoch, epoch_iter = MakeRecordFloder(name, time_now, opt, filename_main, opt.load_pretrain, saveroot)
    train_metrics = setFigureDA()
    val_metrics = setFigureval()
    T_metrics = setFigureval()
    figure_train_metrics = train_metrics.initialize_figure()
    figure_val_metrics = val_metrics.initialize_figure()
    figure_T_metrics = T_metrics.initialize_figure()

    print('\n########## Load the Source Dataset #################')
    opt.phase = 'train'
    train_loader = CreateDataLoader(opt)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(train_loader)))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                              (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(train_loader)) + '\n')
    opt.phase = 'val'
    val_loader = CreateDataLoader(opt)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(val_loader)))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                              (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(val_loader)) + '\n')
    # opt.phase = 'background'
    # background_loader = CreateDataLoader(opt)
    # print("[%s] dataset [%s] was created successfully! Num= %d" %
    #       (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(background_loader)))
    # cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
    #                           (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(background_loader)) + '\n')

    print('\n########## Load the Target Dataset #################')
    t_loaderDict = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
            opt.t = i
            opt.phase = 'target'
            t_loader = CreateDataLoader(opt)
            t_loaderDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)) + '\n')
    print(t_loaderDict)

    t_loaderTestDict = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
            opt.t = i
            opt.phase = 'targetTest'
            t_loader = CreateDataLoader(opt)
            t_loaderTestDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)) + '\n')
    print(t_loaderTestDict)
    tool = CDModelutil(opt)

    print('\n########## Build the Molde #################')
    # initialize model
    model_state_dict = None
    opt.phase = 'train'
    # kernelsize=17
    # net=FCSiamDiffMaskAttentionCenterMask(device=DEVICE,kernelsize=kernelsize,B=opt.batch_size)
    # net = FCSiamDiffMaskAttention()
    if opt.load_pretrain:
        _, _, _, _, Centerdict = tool.load_ckptGANCenter(save_path)
        unchgN = Centerdict[1]
        chgN = Centerdict[2]
    else:
        unchgN = 5
        chgN = 5
    net=FCSiamDiffMaskAttentionCenter(unchgN=unchgN,chgN=chgN)
    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(DEVICE)
    model_D1 = FCDiscriminatorHighMask(num_classes=2)
    model_D1.to(DEVICE)
    model_D2 = FCDiscriminatorLowMask(num_classes=256)
    model_D2.to(DEVICE)

    print('DEVICE:', DEVICE)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr,
                                     betas=(0.5, 0.999))
    else:
        raise NotImplemented(opt.optimizer)
    LEARNING_RATE_D = 1e-4
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    if opt.load_pretrain:
        modelL_state_dict, modelGAN_state_dict, modelGAN2_state_dict, optimizer_state_dict,Centerdict = tool.load_ckptGANCenter(save_path)#,Centerdict
        if modelL_state_dict is not None:
            model_utils.init_weights(net, modelL_state_dict, None, False)
        if modelGAN_state_dict is not None:
            model_utils.init_weights(model_D1, modelGAN_state_dict, None, False)
        if modelGAN2_state_dict is not None:
            model_utils.init_weights(model_D2, modelGAN2_state_dict, None, False)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        if Centerdict is not None:
            unchgN=Centerdict[1]
            chgN=Centerdict[2]
            Center=Centerdict[0]
            print('Center',Center.shape,unchgN,chgN)
            CenterSingel = torch.cat([Center[:, :, :unchgN].mean(dim=2), Center[:, :, chgN:].mean(dim=2)],
                                     dim=-1).unsqueeze(-1)  # ([14, 32, 2, 1])
    else:
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}




    print('optimizer:', opt.optimizer)
    cfg.TRAINLOG.LOGTXT.write('optimizer: ' + opt.optimizer + '\n')

    visualizer = Visualizer(opt)
    tmp = 1
    running_metric = ConfuseMatrixMeter(n_class=2)
    TRAIN_ACC = np.array([], np.float32)
    VAL_ACC = np.array([], np.float32)
    best_val_acc = 0.0
    best_epoch = 0
    # mse = torch.nn.MSELoss(reduction='mean')
    # config GAN

    gan = 'Vanilla'
    bce_loss = torch.nn.functional.binary_cross_entropy

    source_label = 0
    target_label = 1
    DA = True
    TagerMetricNum = 0
    retinex_synthesisModel=retinex_synthesis(device=DEVICE).to(DEVICE)
    zerosLabel = torch.zeros((1, 1, 256, 256)).to(DEVICE)
    cm = plt.cm.get_cmap('jet')
    AllDataMaskGet = pseudoMultiCenter(orisize=256, device=DEVICE, kernelsize=17, B=opt.batch_size).to(
        DEVICE)  # genMaskPatch genpatch
    bin_edgesunchg = torch.linspace(0, 0.5, steps=1000).to(DEVICE)  # 假设概率在0到1之间
    bin_edgeschg = torch.linspace(0, 0.5, steps=1000).to(DEVICE)  # 假设概率在0到1之间
    histogramunchg = None
    histogramchg = None
    MEntropyunchgT = None
    MEntropychgT = None
    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        # if not (epoch > Nepoch or opt.load_pretrain) or (epoch == 1 and opt.load_pretrain) :
        if False:
            opt.phase = 'val'
            t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[1]]
            train_data = t_loader.load_data()
            # train_data = train_loader.load_data()
            iter_source = iter(train_data)
            train_data_len = len(train_data)
            len_source_loader = train_data_len
            tbar = tqdm(range(train_data_len))
            unchgCenter = np.zeros((32, 1), dtype=np.float64)
            chgCenter = np.zeros((32, 1), dtype=np.float64)
            unchgCenter = torch.tensor(unchgCenter, requires_grad=False).to(DEVICE)
            chgCenter = torch.tensor(chgCenter, requires_grad=False).to(DEVICE)
            unchgCenternp = []
            chgCenternp = []
            net.eval()
            tC = time.time()
            centerCur=torch.zeros(32,2).to(DEVICE)
            with torch.no_grad():
                for i in tbar:
                    data = next(iter_source)
                    epoch_iter += opt.batch_size
                    ST1 = Variable(data['t1_img']).to(DEVICE)
                    ST2 = Variable(data['t2_img']).to(DEVICE)
                    labelS = Variable(data['label']).to(DEVICE)
                    cd_predS, defeat3S, defeatSDA = net.forward(ST1, ST2)
                    centerCur = tool.getCenterS2(defeatSDA, labelS.long(),centroidsLast=centerCur,device= DEVICE)
                    unchgCenternp.append(centerCur[:,0].unsqueeze(0))
                    chgCenternp.append(centerCur[:,1].unsqueeze(0))
                    if i > 50 and ttest:
                        break
                #######sklearn

                unchgCenternp = torch.cat(unchgCenternp, dim=0)  # [750, 32, 1]
                unchgCenterMean=unchgCenternp.mean(dim=0).unsqueeze(0)
                # print('unchgCenternp',unchgCenternp.shape,(unchgCenternp.mean(dim=0)).shape)
                # print('unchgCenternp',unchgCenternp.shape)
                unchgCenternp = unchgCenternp.detach().cpu().numpy()
                unchgcluster = KMeans(n_clusters=unchgN, random_state=0).fit(unchgCenternp)

                chgCenternp = torch.cat(chgCenternp, dim=0)  # [750, 32, 1]
                chgCenterMean=chgCenternp.mean(dim=0).unsqueeze(0)

                chgCenternp = chgCenternp.detach().cpu().numpy()
                chgcluster = KMeans(n_clusters=chgN, random_state=0).fit(chgCenternp)

                Center = np.concatenate([unchgcluster.cluster_centers_, chgcluster.cluster_centers_], axis=0)  # (2, 32)
                Center = torch.Tensor(Center).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
                Center = Center.repeat(opt.batch_size, 1, 1).unsqueeze(-1)  # torch.Size([14, 32, 10, 1])

                # CenterSingel=torch.cat([unchgCenternp.mean(dim=1),chgCenternp.mean(dim=1)],dim=)
                CenterSingel=torch.cat([Center[:,:,:unchgN].mean(dim=2),Center[:,:,unchgN:].mean(dim=2)],dim=-1).unsqueeze(-1)#([14, 32, 2, 1])

                print(epoch,': Generated Center!',Center.shape,CenterSingel.shape)
        ####################################val
        running_metric.clear()
        opt.phase = 'val'
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        val_dataload = val_loader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        tbar = tqdm(range(val_data_len))
        chgCenterS = []
        unchgCenterS = []
        with torch.no_grad():
            net.eval()
            model_D1.eval()
            model_D2.eval()
            for i in tbar:
                data_val = next(iter_val)
                data_T1_val = Variable(data_val['t1_img']).to(DEVICE)
                data_T2_val = Variable(data_val['t2_img']).to(DEVICE)
                label = Variable(data_val['label']).to(DEVICE)
                if epoch > Nepoch or opt.load_pretrain:
                    cd_val_pred = net.forward(data_T1_val, data_T2_val, DomainLabel=1, Scenter=Center.detach())
                else:
                    cd_val_pred = net.forward(data_T1_val, data_T2_val,DomainLabel=0)
                centerSchg=(cd_val_pred[-1]*label).sum(dim=[0,2,3])/label.sum()
                centerSunchg=(cd_val_pred[-1]*(1-label)).sum(dim=[0,2,3])/(1-label).sum()
                # print(centerSunchg.shape)
                chgCenterS.append(centerSchg.cpu().numpy())
                unchgCenterS.append(centerSunchg.cpu().numpy())

                TCELoss = cross_entropy(cd_val_pred[0], label.long())
                record['SCET'] += TCELoss.item()
                record['LossT'] = record['SCET']
                # update metric
                Score = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                val_data = data_val['label'].detach()
                val_pred = torch.argmax(F.softmax(cd_val_pred[0], dim=1), dim=1)
                current_score = running_metric.confuseMS(pr=val_pred.cpu().numpy(), gt=val_data.cpu().numpy())  # 更新
                Score.update(current_score)
                valMessage = visualizer.print_current_scores(opt.phase, epoch, i, val_data_len, Score)
                tbar.set_description(valMessage)
                if i > 10 and ttest:
                    break

        chgArray = np.array(chgCenterS)
        unchgArray = np.array(unchgCenterS)

        # print(chgArray.shape,unchgArray.shape)
        one = np.ones(chgArray.shape[0])
        zero = np.zeros(chgArray.shape[0])
        labelSS = np.concatenate([one, zero], axis=0)
        featureS = np.concatenate([chgArray, unchgArray], axis=0)
        vis = FeatureVisualize()
        # vis.plot_tsne(features=feature, labels=label)


        val_scores = running_metric.get_scores()
        # visualizer.print_scores(opt.phase, epoch, val_scores)

        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        IterValScore = {'Loss': lossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = visualizer.print_scores(opt.phase, epoch, IterValScore)

        cfg.TRAINLOG.LOGTXT.write(message + '\n')

        exel_out = opt.phase, epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, IterValScore['acc'], \
                   IterValScore['unchgAcc'], IterValScore['chgAcc'], IterValScore['recall_1'], \
                   IterValScore['F1_1'], IterValScore['miou'], IterValScore['precision_1'], \
                   str(IterValScore['tn']), str(IterValScore['tp']), str(IterValScore['fn']), str(
            IterValScore['fp'])

        figure_val_metrics = val_metrics.set_figure(metric_dict=figure_val_metrics,
                                                    nochange_acc=IterValScore['unchgAcc'],
                                                    change_acc=IterValScore['chgAcc'],
                                                    prec=IterValScore['precision'], rec=IterValScore['recall'],
                                                    f_meas=IterValScore['mf1'], total_acc=IterValScore['acc'],
                                                    Iou=IterValScore['miou'], CES_lossAvg=CES_lossAvg)
        cfg.TRAINLOG.EXCEL_LOG['wsVal'].append(exel_out)

        val_epoch_acc = val_scores['acc']
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch
        # print('Center',Center.shape)
        if saveLast:
            if epoch == opt.num_epochs:
                save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                           % (
                               name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'],
                               val_scores['unchgAcc'])
                # tool.save_cd_ckpt(iters=epoch, network=net, optimizer=optimizer, save_str=save_str)
                # tool.save_ckptGAN(network=[net,model_D1,model_D2], optimizer=optimizer, save_str=save_str)
                tool.save_ckptGANCenter(network=[net,model_D1,model_D2],Center=[Center,unchgN,chgN], optimizer=optimizer, save_str=save_str)


        else:
            save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                       % (name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])

            # tool.save_cd_ckpt(iters=epoch,network=net,optimizer=optimizer,save_str=save_str)
            # tool.save_ckptGANCenter(network=[net,model_D1,model_D2], optimizer=optimizer, save_str=save_str)
            tool.save_ckptGANCenter(network=[net, model_D1, model_D2], Center=[Center,unchgN,chgN], optimizer=optimizer,
                                    save_str=save_str)

        save_pickle(figure_train_metrics, "./log/%s/%s/fig_train.pkl" % (name, time_now))
        save_pickle(figure_val_metrics, "./log/%s/%s/fig_val.pkl" % (name, time_now))

        # end of epoch
        # print(opt.num_epochs,opt.num_decay_epochs)
        iter_end_time = time.time()
        # print('End of epoch %d / %d \t Time Taken: %d sec \t best acc: %.5f (at epoch: %d) ' %
        #       (epoch, opt.num_epochs + opt.num_decay_epochs, time.time() - epoch_start_time, best_val_acc, best_epoch))
        # np.savetxt(cfg.TRAINLOG.ITER_PATH, (epoch + 1, 0), delimiter=',', fmt='%d')
        cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))

        # cfg.TRAINLOG.LOGTXT.write('\n================ Target Test (%s) ================\n' % time.strftime("%c"))
        if epoch % 1 == 0:
            # print('================epoch:%d Target Test (%s) ================\n' % (epoch, time.strftime("%c")))
            # cfg.TRAINLOG.LOGTXT.write(
            #     '\n================epoch:%d Target Test (%s) ================\n' % (epoch, time.strftime("%c")))
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
                TagerMetricNum += 1

                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                t_data = t_loader.load_data()
                t_data_len = len(t_data)
                tbar = tqdm(range(t_data_len))
                iter_t = iter(t_data)
                running_metric.clear()
                opt.phase = 'target'
                # MEntropy=[]
                # bin_edgesunchg = torch.linspace(0, 0.5, steps=1000).to(DEVICE)  # 假设概率在0到1之间
                # bin_edgeschg = torch.linspace(0, 0.5, steps=1000).to(DEVICE)  # 假设概率在0到1之间
                # histogramunchg = None
                # histogramchg=None
                # histogramunchg = []
                # histogramchg = []
                chgCenternp=[]
                unchgCenternp=[]
                with torch.no_grad():
                    net.eval()
                    model_D1.eval()
                    model_D2.eval()
                    record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
                    for i in tbar:
                        data_test = next(iter_t)
                        data_T1_test = Variable(data_test['t1_img']).to(DEVICE)
                        data_T2_test = Variable(data_test['t2_img']).to(DEVICE)
                        labelT = Variable(data_test['label']).to(DEVICE)
                        if epoch > Nepoch or opt.load_pretrain:
                            cd_target_pred = net.forward(data_T1_test, data_T2_test, DomainLabel=1, Scenter=Center.detach())
                        else:
                            cd_target_pred = net.forward(data_T1_test, data_T2_test, DomainLabel=0)
                        centerSchg = (cd_target_pred[-1] * labelT).sum(dim=[0, 2, 3]) / (labelT.sum()+0.000001)
                        centerSunchg = (cd_target_pred[-1] * (1 - labelT)).sum(dim=[0, 2, 3]) / ((1 - labelT).sum()+0.000001)
                        # print(centerSunchg.shape)
                        # chgCenterT.append(centerSchg.cpu().numpy())
                        # unchgCenterT.append(centerSunchg.cpu().numpy())
                        # centerCur = tool.getCenterS2(cd_target_pred[-1], labelT.long(), centroidsLast=centerCur, device=DEVICE)

                        unchgCenternp.append(centerSchg.unsqueeze(0))
                        chgCenternp.append(centerSunchg.unsqueeze(0))
                        # curcentersT = tool.getCenterSmulti(cd_target_pred[-1].detach(), labelT.detach().long(),
                        #                                    centroidsLast=CenterSingel[0, :, :, 0],
                        #                                    device=DEVICE)  # ([2, 32, 14])
                        #
                        # print('curcentersT',curcentersT.shape,centerCur[:, 0].shape)
                        # unchgCenternp.append(centerSchg[0, :,0].unsqueeze(0))
                        # chgCenternp.append(curcentersT[1, :,0].unsqueeze(0))
                        # cd_target_pred = net.forward(data_T1_test, data_T2_test)
                        # cd_val_loss, cd_val_pred = cd_model(data['t1_img'].cuda(), data['t2_img'].cuda(), data['label'].cuda())
                        # TCELoss = tool.loss(cd_val_pred[0], label.long())
                        TCELoss = cross_entropy(cd_target_pred[0].detach(), labelT.long())

                        record['TCET'] += TCELoss.item()
                        record['LossT'] = record['TCET']
                        # update metric
                        Score = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                        target_pred = torch.argmax(cd_target_pred[0].detach(), dim=1)
                        SelectMaskOut = AllDataMaskGet.forward(
                            targetimg={'img1': data_T1_test, 'img2': data_T2_test, 'labelT': labelT, 'i': i,
                                       'Pseudo': target_pred.unsqueeze(1),
                                       'defeatTDA': cd_target_pred[-1], 'UnchgN': unchgN, 'MEntropyunchgT': MEntropyunchgT,
                                        'MEntropychgT': MEntropychgT, 'epoch': epoch},
                            infeat=cd_target_pred[0], multiCenter=Center)
                        entropyunchg = SelectMaskOut[2][:, 0, :, :].reshape(-1)
                        entropychg = SelectMaskOut[2][:, 1, :, :].reshape(-1)

                        histunchg = torch.histc(entropyunchg[entropyunchg > 0.01], bins=1000, min=0, max=0.5)
                        histchg = torch.histc(entropychg[entropychg > 0.01], bins=1000, min=0, max=0.5)
                        if histogramunchg is None:
                            histogramunchg = histunchg
                        else:
                            histogramunchg += histunchg
                        if histogramchg is None:
                            histogramchg = histchg
                        else:
                            histogramchg += histchg
                        # histogramunchg.extend(entropyunchg)
                        # histogramchg.extend(entropychg)
                        pseudo = torch.tensor(SelectMaskOut[0].detach(), dtype=torch.int).requires_grad_(False)
                        current_scoreT = running_metric.confuseMT(pr=pseudo.cpu().numpy(), gt=labelT.cpu().numpy())

                        # current_scoreT = running_metric.confuseMT(pr=target_pred.cpu().numpy(),
                        #                                          gt=labelT.cpu().numpy())  # 更新
                        Score.update(current_scoreT)
                        valMessageT = visualizer.print_current_scores(opt.phase, epoch, i, t_data_len, Score)
                        tbar.set_description(valMessageT)
                        if i > 10 and ttest:
                            break

                    unchgCenternp = torch.cat(unchgCenternp, dim=0)  # [750, 32, 1]
                    unchgCenterMean = unchgCenternp.mean(dim=0).unsqueeze(0)

                    # print('unchgCenternp',unchgCenternp.shape,(unchgCenternp.mean(dim=0)).shape)
                    # print('unchgCenternp',unchgCenternp.shape)
                    unchgCenternp = unchgCenternp.detach().cpu().numpy()
                    unchgcluster = KMeans(n_clusters=unchgN, random_state=0).fit(unchgCenternp)

                    chgCenternp = torch.cat(chgCenternp, dim=0)  # [750, 32, 1]
                    chgCenterMean = chgCenternp.mean(dim=0).unsqueeze(0)
                    print('chgCenterMean',chgCenterMean.shape,chgCenternp.shape,unchgCenternp.shape)
                    chgCenternp = chgCenternp.detach().cpu().numpy()
                    chgcluster = KMeans(n_clusters=chgN, random_state=0).fit(chgCenternp)

                    Center = np.concatenate([unchgcluster.cluster_centers_, chgcluster.cluster_centers_],
                                            axis=0)  # (2, 32)
                    Center = torch.Tensor(Center).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
                    Center = Center.repeat(opt.batch_size, 1, 1).unsqueeze(-1)  # torch.Size([14, 32, 10, 1])

                    # CenterSingel=torch.cat([unchgCenternp.mean(dim=1),chgCenternp.mean(dim=1)],dim=)
                    CenterSingel = torch.cat([Center[:, :, :unchgN].mean(dim=2), Center[:, :, unchgN:].mean(dim=2)],
                                             dim=-1).unsqueeze(-1)  # ([14, 32, 2, 1])


                    CenterArrayunchg=np.array(Center[0,:,:5].reshape(-1,32).cpu().numpy())
                    CenterArraychg=np.array(Center[0,:,5:].reshape(-1,32).cpu().numpy())

                    print('chgArray',chgArray.shape,Center.shape,CenterArraychg.shape)
                    # print(chgArray.shape,unchgArray.shape)
                    one = np.ones(chgArray.shape[0])
                    zero = np.zeros(chgArray.shape[0])
                    oneCenter=np.ones(CenterArrayunchg.shape[0])
                    oneCenterSign = np.ones(chgCenterMean.shape[0])
                    # label = np.concatenate([one, zero,oneCenter*2,oneCenter*3,oneCenterSign*4,oneCenterSign*5], axis=0)
                    # featureT = np.concatenate([chgArray, unchgArray,CenterArrayunchg,CenterArraychg,
                    #                            unchgCenterMean.cpu().numpy(),chgCenterMean.cpu().numpy()], axis=0)

                    label = np.concatenate(
                        [one, zero], axis=0)
                    featureT = np.concatenate([chgArray, unchgArray], axis=0)
                    feature=np.concatenate([featureS,featureT],axis=0)
                    # label=np.concatenate([labelSS,one*2,one*3],axis=0)
                    vis = FeatureVisualize()
                    vis.plot_tsne(features=featureT, labels=label)



                    ttt1=time.time()
                    cumulative_distributionunchg = torch.cumsum(histogramunchg, dim=0)
                    # print(cumulative_distributionunchg.shape, histogramunchg.shape)  # torch.Size([1000]) torch.Size([1000])
                    total_countunchg = cumulative_distributionunchg[-1]
                    # 查找第20%的概率值
                    threshold_indexunchg = torch.searchsorted(cumulative_distributionunchg,
                                                              (0.1 + 0.05 * epoch) * total_countunchg)
                    MEntropyunchgT = bin_edgesunchg[threshold_indexunchg].item()
                    # print('total_countunchg', total_countunchg)
                    ############chg
                    # print('histogramchg',histogramchg.shape)
                    cumulative_distributionchg = torch.cumsum(histogramchg, dim=0)
                    total_countchg = cumulative_distributionchg[-1]
                    # 查找第20%的概率值
                    threshold_indexchg = torch.searchsorted(cumulative_distributionchg,
                                                            (0.1 + 0.05 * epoch) * total_countchg)
                    MEntropychgT = bin_edgeschg[threshold_indexchg].item()
                    # print('total_countchg', threshold_indexunchg, threshold_indexchg)
                    print('Threshold out:', MEntropyunchgT, MEntropychgT)


                    target_scores = running_metric.get_scoresTT()
                    lossAvg = record['LossT'] / i
                    CES_lossAvg = record['SCET'] / i
                    DGAN_LossAvg = record['DGANT'] / i
                    CET_lossAvg = record['TCET'] / i
                    IterTargetScore = {'Loss': lossAvg, 'TCE': CET_lossAvg}
                    IterTargetScore.update(target_scores)
                    message = visualizer.print_scores('target out', epoch, IterTargetScore)
                    cfg.TRAINLOG.LOGTXT.write('\n================ [%s] Target Test (%s) ================\n' % (
                        cfg.TRAINLOG.DATA_NAMES[kk], time.strftime("%c")))

                    cfg.TRAINLOG.LOGTXT.write(message + '\n')

                    exel_out = 'T-' + cfg.TRAINLOG.DATA_NAMES[kk][
                        0], epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, IterTargetScore['acc'], \
                               IterTargetScore['unchgAcc'], val_scores['chgAcc'], IterTargetScore['recall'], \
                               IterTargetScore['mf1'], val_scores['miou'], IterTargetScore['precision'], \
                               str(IterTargetScore['tn']), str(IterTargetScore['tp']), str(IterTargetScore['fn']), str(
                        IterTargetScore['fp'])

                    # cfg.TRAINLOG.EXCEL_LOG['wsT-' + cfg.TRAINLOG.DATA_NAMES[kk][0]].append(exel_out)
                    # figure_T_metrics = T_metrics.set_figure(metric_dict=figure_T_metrics,
                    #                                         nochange_acc=IterTargetScore['unchgAcc'],
                    #                                         change_acc=IterTargetScore['chgAcc'],
                    #                                         prec=IterTargetScore['precision'],
                    #                                         rec=IterTargetScore['recall'],
                    #                                         f_meas=IterTargetScore['mf1'],
                    #                                         total_acc=IterTargetScore['acc'],
                    #                                         Iou=IterTargetScore['miou'], CES_lossAvg=CET_lossAvg)
                    # save_pickle(figure_T_metrics, "./log/%s/%s/fig_T.pkl" % (name, time_now))
                    cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
                # cfg.TRAINLOG.EXCEL_LOG.get_sheet_by_name(cfg.TRAINLOG.EXCEL_LOGSheet[kk]).append(exel_out)
    print('================ Training Completed (%s) ================\n' % time.strftime("%c"))
    cfg.TRAINLOG.LOGTXT.write('\n================ Training Completed (%s) ================\n' % time.strftime("%c"))
    plotFigureDA(figure_train_metrics, figure_val_metrics, opt.num_epochs + opt.num_decay_epochs, name, opt.model_type,
                 time_now)
    plotFigureTarget(figure_T_metrics, TagerMetricNum, name, opt.model_type,
                     time_now)  #
    time_end = time.strftime("%Y%m%d-%H_%M", time.localtime())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # if scheduler:
    #     print('Training Start lr:', lr, '  Training Completion lr:', scheduler.get_last_lr())
    print('Training Start Time:', cfg.TRAINLOG.STARTTIME, '  Training Completion Time:', time_end, '  Total Epoch Num:',
          epoch)
    print('saved path:', './log/{}/{}'.format(name, time_now))
    cfg.TRAINLOG.LOGTXT.write(
        'Training Start Time:' + cfg.TRAINLOG.STARTTIME + '  Training Completion Time:' + time_end + 'Total Epoch Num:' + str(
            epoch) + '\n')
