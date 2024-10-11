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
from model.Siamese_diff import FCSiamDiffMaskAttention,FCSiamDiffMaskAttentionCenter
import matplotlib.pyplot as plt
import time
import numpy as np
import os


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
def CORAL(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)#maskunchg
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)
    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4 * d * d)
    return loss

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
    saveLast = False
    name = opt.dset + 'CDPretrain2/' + opt.model_type + 'Experimnet'

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
        # cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch3', 'GZ_CDPatchNoResolution']

        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatchNoResolution', 'LEVIR_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'WHBuildingNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuilding3', 'GDbuilding']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'S2LookingPatchNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['GDbuilding', 'WHBuildingNoUnchg']
        cfg.TRAINLOG.DATA_NAMES = [ 'WHBuildingNoUnchg','GDbuilding']
    Nepoch = 10

    opt.load_pretrain = True
    if opt.load_pretrain:
        # saveroot = './log/CD_DA_buildingCDPretrain3/2-3CenterAttenEntropyMidGAN-itermultiCenter-Dist/20231203-17_04_L-GZ-Th060-weight-STkmean-paste/'
        # save_path = saveroot + '/savemodel/_31_acc-0.9825_chgAcc-0.8235_unchgAcc-0.9894.pth'
        saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-Dist/20231201-10_27_G-WH-paste3-iter-NoAdjust_Th015aug-060/'
        save_path = saveroot + '/savemodel/_31_acc-0.9890_chgAcc-0.8799_unchgAcc-0.9937.pth'
        # saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-Dist/20231201-10_27_W-GD-paste3-iter-NoAdjust_Th015aug-060/'
        # save_path = saveroot + '/savemodel/_31_acc-0.9692_chgAcc-0.8889_unchgAcc-0.9725.pth'
        opt.num_epochs = 20
        Nepoch=1
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
        "%Y%m%d-%H_%M_" + cfg.TRAINLOG.DATA_NAMES[opt.s][0] + '-' + cfg.TRAINLOG.DATA_NAMES[opt.t][:2] + '-test',
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
    kernelsize=7
    MeanKernal = torch.ones(1, 1, kernelsize, kernelsize, device=DEVICE) / (kernelsize * kernelsize)
    MeanKernal = nn.Parameter(MeanKernal, requires_grad=False)

    entropyT=0
    entropyThrOut={'entropymeanValue': [], 'entropyvarValue': [], 'entropyunchgValue': [], 'entropychgValue': []}

    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        # if (epoch<10 and not opt.load_pretrain) or (opt.load_pretrain and epoch==1) :
        # if not (epoch > Nepoch or opt.load_pretrain) or (epoch == 1 and opt.load_pretrain) :
        if False:
            opt.phase = 'val'
            train_data = train_loader.load_data()
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
                # print('unchgCenternp',unchgCenternp.shape)
                unchgCenternp = unchgCenternp.detach().cpu().numpy()
                unchgcluster = KMeans(n_clusters=unchgN, random_state=0).fit(unchgCenternp)

                chgCenternp = torch.cat(chgCenternp, dim=0)  # [750, 32, 1]
                chgCenternp = chgCenternp.detach().cpu().numpy()
                chgcluster = KMeans(n_clusters=chgN, random_state=0).fit(chgCenternp)

                Center = np.concatenate([unchgcluster.cluster_centers_, chgcluster.cluster_centers_], axis=0)  # (2, 32)
                Center = torch.Tensor(Center).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
                Center = Center.repeat(opt.batch_size, 1, 1).unsqueeze(-1)  # torch.Size([14, 32, 10, 1])

                CenterSingel=torch.cat([Center[:,:,:unchgN].mean(dim=2),Center[:,:,unchgN:].mean(dim=2)],dim=-1).unsqueeze(-1)#([14, 32, 2, 1])

                print(epoch,': Generated Center!',Center.shape,CenterSingel.shape)

        opt.phase = 'train'
        #Load Data
        train_data = train_loader.load_data()
        iter_source = iter(train_data)
        train_data_len = len(train_data)
        len_source_loader = train_data_len
        #Load Data
        # background_dataload = background_loader.load_data()
        # iter_background = iter(background_dataload)
        # len_background_loader = len(background_dataload)
        #Load Data
        Tt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        len_target_loader = len(Tt_data)
        iter_target = iter(Tt_data)
        #init
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % len(train_loader)
        running_metric.clear()
        net.train()
        model_D1.train()
        model_D2.train()
        tbar = tqdm(range(len_target_loader - 1))
        train_data_len = len_target_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        # adjust_learning_rate(optimizer, epoch, opt.num_epochs+1)
        # adjust_learning_rate_D(optimizer_D1, epoch, opt.num_epochs+1)
        # adjust_learning_rate_D(optimizer_D2, epoch, opt.num_epochs+1)
        entropylist=[]
        varentropylist=[]
        entropyThrList = {'entropymean': [], 'entropyvar': [], 'entropyunchg': [], 'entropychg': []}

        for i in tbar:
            try:
                Sdata = next(iter_source)
            except:
                iter_source = iter(train_data)
            try:
                Tdata = next(iter_target)
            except:
                iter_target = iter(Tt_data)
            epoch_iter += opt.batch_size

            ############## Forward Pass ######################
            ST1 = Variable(Sdata['t1_img']).to(DEVICE)
            ST2 = Variable(Sdata['t2_img']).to(DEVICE)
            labelS = Variable(Sdata['label']).to(DEVICE)
            TT1 = Variable(Tdata['t1_img']).to(DEVICE)
            TT2 = Variable(Tdata['t2_img']).to(DEVICE)
            labelT = Variable(Tdata['label']).to(DEVICE)

            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            # Generation: source
            # optimizer.zero_grad()
            # optimizer_D1.zero_grad()
            # optimizer_D2.zero_grad()
            # if epoch<10 or opt.load_pretrain:
            if epoch > Nepoch or opt.load_pretrain:
                # cd_predS, defeat3S, defeatSDA = net.forward(ST1, ST2, DomainLabel=1, Scenter=CenterSingel.detach())
                # cd_predT, defeat3T, defeatTDA = net.forward(TT1, TT2, DomainLabel=1, Scenter=CenterSingel.detach())
                cd_predS, defeat3S, defeatSDA = net.forward(ST1, ST2, DomainLabel=1, Scenter=Center.detach())
                cd_predT, defeat3T, defeatTDA = net.forward(TT1, TT2, DomainLabel=1, Scenter=Center.detach())
            else:
                cd_predS, defeat3S, defeatSDA = net.forward(ST1, ST2, DomainLabel=0)
                cd_predT, defeat3T, defeatTDA = net.forward(TT1, TT2, DomainLabel=0)

            # cd_predT, defeat3T, defeatTDA = net(TT1, TT2)
            # cd_predT, defeat3T, defeatTDA = net.forward(TT1, TT2,DomainLabel=1,Scenter=Center)

            SCELoss = cross_entropy(cd_predS, labelS.long())
            # SCELoss.backward()
            record['SCET'] += SCELoss.item()

            # predict
            cd_predSo = torch.argmax(cd_predS.detach(), dim=1)
            softmaxpreT = F.softmax(cd_predT, dim=1).detach()
            Tout = torch.argmax(softmaxpreT.detach(), dim=1)
            labelRatio = labelS.sum(dim=(1, 2, 3)) / (labelS.size(2) * labelS.size(3))
            PseudoRatio = Tout.sum(dim=(1, 2)) / (Tout.size(1) * Tout.size(2))
            Entropy = (-softmaxpreT * torch.log2(softmaxpreT + 1e-10)).detach()
            EntropyVar = Entropy[:, 0, :, :].var(dim=[1, 2])
            EntropyM = Entropy[:, 0, :, :].mean(dim=[1, 2])  # lower better
            entropyThrList['entropymean'].append(EntropyM)
            entropyThrList['entropyvar'].append(EntropyVar)

            EntropyunchgT = (Entropy[:, 0, :, :] * (1 - Tout)).sum(dim=[1, 2]) / ((1 - Tout).sum())
            entropyThrList['entropyunchg'].append(EntropyunchgT)
            EntropychgT = (Entropy[:, 1, :, :] * (Tout)).sum(dim=[1, 2]) / ((Tout).sum())
            entropyThrList['entropychg'].append(EntropychgT)


            DistLoss = torch.tensor([0.0]).to(DEVICE)
            if (epoch > Nepoch):
                for cc in range(labelT.size(0)):
                    if (1-Tout[cc]).sum() !=0 and Tout[cc].sum()!=0:
                        if EntropyunchgT[cc]<entropyThrOut['entropyunchgValue'] or EntropychgT[cc]<entropyThrOut['entropychgValue']:
                            zeross=torch.zeros_like(Tout[cc]).to(DEVICE)
                            oness=torch.ones_like(Tout[cc]).to(DEVICE)
                            # print('maskunchg',maskunchg.shape,Entropy.shape)
                            maskunchg=torch.where(Entropy[cc,0,:,:]<entropyThrOut['entropyunchgValue'],oness,zeross)
                            maskchg=torch.where(Entropy[cc,1,:,:]<entropyThrOut['entropychgValue'],oness,zeross)
                            # maskunchg[Entropy[cc,0,:,:]<entropyThrOut['entropyunchgValue']]=1
                            # maskchg[Entropy[cc,1,:,:]<entropyThrOut['entropychgValue']]=1

                            if ((maskunchg * (1 - Tout[cc])).sum()) > 0:
                                # distchg = torch.sum((Center[0, :, chgN:, 0] - CenterchgT.unsqueeze(-1)) ** 2, dim=0)
                                CenterUnchgT = (defeatTDA[cc] * maskunchg * (1 - Tout[cc].detach())).sum(dim=[1, 2]) / (
                                    (maskunchg * (1 - Tout[cc].detach())).sum())
                                distUnchg = torch.sum((Center[0, :, :unchgN, 0] - CenterUnchgT.unsqueeze(-1)) ** 2,
                                                      dim=0)  # [5]
                                nearest_centroidsunchg = torch.argmin(distUnchg, dim=0)
                                # distUnchgOut=(((maskunchg.detach()*(1-Tout[cc])).sum())/((1-Tout[cc]).sum()))*distUnchg[nearest_centroidsunchg]
                                coralunchgdist = (((maskunchg.detach() * (1 - Tout[cc].detach())).sum()) / (
                                    (1 - Tout[cc].detach()).sum())) * \
                                                 CORAL(Center[0, :, nearest_centroidsunchg, 0].unsqueeze(-1),
                                                       CenterUnchgT.unsqueeze(-1))
                                DistLoss = DistLoss + coralunchgdist
                            if ((maskchg * (Tout[cc])).sum()) > 0:
                                CenterchgT = (defeatTDA[cc] * maskchg * (Tout[cc].detach())).sum(dim=[1, 2]) / (
                                    (maskchg * (Tout[cc].detach())).sum())
                                distchg = torch.sum((Center[0, :, chgN:, 0].detach() - CenterchgT.unsqueeze(-1)) ** 2,
                                                    dim=0)
                                nearest_centroidschg = torch.argmin(distchg, dim=0)
                                # distchgOut = (((maskchg.detach()*Tout[cc]).sum())/(Tout[cc].sum()))*distchg[nearest_centroidschg]
                                coralchgdist = (((maskchg.detach() * Tout[cc].detach()).sum()) / (Tout[cc].detach().sum())) * \
                                               CORAL(Center[0, :, chgN + nearest_centroidschg, 0].unsqueeze(-1),
                                                     CenterchgT.unsqueeze(-1))
                                DistLoss = DistLoss + coralchgdist

            D_out1 = model_D1(prob_2_entropy(F.softmax(cd_predT, dim=1)))
            D_out2 = model_D2(defeat3T['outF'])
            loss_GT = 0.001 * bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).to(DEVICE))
            loss_GT2 = 0.001 * bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).to(DEVICE))
            # (loss_GT + loss_GT2 + SCELoss + DistLoss).backward()


            #############################Discrimination##################################
            # train with source
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
            D_out1S = model_D1(prob_2_entropy(F.softmax(cd_predS, dim=1)).detach())
            loss_DS = bce_loss(D_out1S, Variable(torch.FloatTensor(D_out1S.data.size()).fill_(source_label)).to(DEVICE))
            record['DGANT'] += loss_DS.item()
            # loss_DS.backward()
            # train with target
            D_out1T = model_D1(prob_2_entropy(F.softmax(cd_predT, dim=1)).detach())
            loss_DT = bce_loss(D_out1T, Variable(torch.FloatTensor(D_out1T.data.size()).fill_(target_label)).to(DEVICE))
            record['DGANT'] += loss_DT.item()
            # loss_DT.backward()

            D_out2S = model_D2(defeat3S['outF'].detach())
            loss_DS2 = bce_loss(D_out2S, Variable(torch.FloatTensor(D_out2S.data.size()).fill_(source_label)).to(DEVICE))
            record['DGANT'] += loss_DS2.item()
            # loss_DS2.backward()
            # train with target
            D_out2T = model_D2(defeat3T['outF'].detach())
            loss_DT2 = bce_loss(D_out2T, Variable(torch.FloatTensor(D_out2T.data.size()).fill_(target_label)).to(DEVICE))
            record['DGANT'] += loss_DT2.item()
            # loss_DT2.backward()
            # optimizer.step()
            # optimizer_D1.step()
            # optimizer_D2.step()

            ####Background
            # optimizer.zero_grad()
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            for param in retinex_synthesisModel.parameters():
                param.requires_grad = False

            TFlag = False
            D = 0
            Bnum = 0
            TCELoss = torch.tensor([0.]).to(DEVICE)
            # if (epoch > 2 or opt.load_pretrain) and True:
            if ((epoch > 0 and opt.load_pretrain) or(epoch > 5 and not opt.load_pretrain)) and True:
            # if True:
                for cc in range(labelS.size(0)):
                    if labelRatio[cc] > 0.1 and labelRatio[cc] < 0.5 and PseudoRatio[cc] < 0.01 and EntropyM[cc] < entropyT:
                        # if labelRatio[cc] > 0.1 and labelRatio[cc] < 0.5 :
                        Bnum += 1
                        TFlag = True
                        BInput = torch.cat(
                            [TT1[cc].unsqueeze(0), TT2[cc].unsqueeze(0)], dim=0)
                        STInput = torch.cat(
                            [ST1[cc].unsqueeze(0), ST2[cc].unsqueeze(0)], dim=0)
                        corrected_imgBS,oout = retinex_synthesisModel.forward(BInput, STInput,draw=True)
                        # print(corrected_imgBS[0].unsqueeze(0).shape,TT1[cc].shape)
                        T1cat = torch.cat([corrected_imgBS[0].unsqueeze(0),
                                           corrected_imgBS[0].unsqueeze(0),
                                           TT1[cc].unsqueeze(0),
                                           # corrected_imgBS[0].unsqueeze(0),
                                           # B1
                                           ], dim=0)
                        T2cat = torch.cat(
                            [(corrected_imgBS[1] * (1 - labelS[cc]) + (ST2[cc] * labelS[cc])).unsqueeze(0),
                                (corrected_imgBS[1] * (1 - 0.5 * labelS[cc]) + 0.5 * ST2[cc] * labelS[cc]).unsqueeze(0),
                                (TT2[cc] * (1 - labelS[cc]) + ST2[cc] * labelS[cc]).unsqueeze(0),
                                # corrected_imgBS[1].unsqueeze(0),
                                # B2
                            ], dim=0)
                        TTlabel = []
                        for ii in range(T2cat.shape[0]):
                            # if ii < 2:
                            TTlabel.append(labelS[cc].unsqueeze(0))
                            # else:
                            #     TTlabel.append(torch.zeros_like(labelS[cc].unsqueeze(0)).to(DEVICE))
                        TTlabel = torch.cat(TTlabel, dim=0)
                        if epoch > Nepoch or opt.load_pretrain:
                            cd_predTT, defeat3TT, defeatTTDA = net.forward(T1cat, T2cat, DomainLabel=1,
                                                                           Scenter=Center.detach())
                        else:
                            cd_predTT, defeat3TT, defeatTTDA = net.forward(T1cat, T2cat, DomainLabel=0)
                        Pro = F.softmax(cd_predTT, dim=1).detach()
                        outputEntropy = -Pro * torch.log2(Pro + 1e-10)  # lower better
                        outputEntropy = nn.functional.conv2d(outputEntropy, MeanKernal.repeat(outputEntropy.shape[1], 1, 1, 1),
                                                            padding=kernelsize // 2, groups=outputEntropy.shape[1])
                        outputEntropy=outputEntropy.sum(1).unsqueeze(1)

                        TCELossa = cross_entropy(cd_predTT, TTlabel.long(), reduction='none')
                        TCELossa = ((1 - outputEntropy) * TCELossa).mean()
                        TCELoss += TCELossa
                        if True:
                            tailor = ST2[cc].unsqueeze(0) * labelS[cc].unsqueeze(0)
                            cd_predT_softmax = F.softmax(cd_predTT, dim=1)

                            titles = ['S1', 'S2','SH1','SH2','SL1','SL2',
                                      'Synthesis T1', 'Synthesis T2','Label',
                                      'T1', 'T2', 'TH1', 'TH2', 'TL1', 'TL2',
                                      'OUTPUT T1', 'OUTPUT T2','tailor'
                                      ]
                            # (corrected_imgBS[1] * (1 - labelS[cc]) + (ST2[cc] * labelS[cc])).unsqueeze(0),
                            # (corrected_imgBS[1] * (1 - 0.5 * labelS[cc]) + 0.5 * ST2[cc] * labelS[cc]).unsqueeze(0),
                            # (TT2[cc] * (1 - labelS[cc]) + ST2[cc] * labelS[cc]).unsqueeze(0),
                            images=[ST1[cc], ST2[cc],oout[2][0],oout[2][1],oout[3][0],oout[3][1],
                                    corrected_imgBS[0],corrected_imgBS[1],labelS[cc],
                                    TT1[cc], TT2[cc],oout[0][0],oout[0][1],oout[1][0],oout[1][1],
                                    (corrected_imgBS[1] * (1 - labelS[cc]) + (ST2[cc] * labelS[cc])),
                                    (corrected_imgBS[1] * (1 - 0.5 * labelS[cc]) + 0.5 * ST2[cc] * labelS[cc]),
                                    tailor[0]]
                            # images = [ST1[cc], ST2[cc],
                            #           corrected_imgBS[0],corrected_imgBS[1],
                            #           cd_predT_softmax[0, 1], labelS[cc],
                            #           TT1[0], TT2[0],
                            #           (corrected_imgBS[1] * (1 - labelS[cc]) + (ST2[cc] * labelS[cc])),
                            #           (corrected_imgBS[1] * (1 - 0.5 * labelS[cc]) + 0.5 * ST2[cc] * labelS[cc]),
                            #           cd_predT_softmax[1, 1], tailor[0]]
                        fig, axes = plt.subplots(2, 9, figsize=(22, 7))  # 假设每张子图为256x256像素，所以整体尺寸为10.24x7.68英寸
                        dN = 0
                        dNn = 0
                        for ax, img in zip(axes.ravel(), images):

                            if len(img.shape) == 3:
                                ax.imshow(img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                                # print(dN, len(titles))
                                ax.set_title(titles[dN])
                                # dN += 1
                            else:
                                # ax.imshow(img, cmap=cm)
                                # ax.set_title(titles[dNn])
                                im = ax.imshow(img.detach().cpu().numpy(), cmap=cm, vmin=0, vmax=1)
                                ax.set_title(titles[dN])
                                divider = make_axes_locatable(ax)
                                cbar_ax = divider.append_axes("right", size="5%", pad=0.15)
                                # cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
                                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                                cbar = fig.colorbar(im, cax=cbar_ax, ticks=ticks)
                                # cbar.ax.set_yticklabels(['0', '1'])  # 设置colorbar标签为0和1
                                cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置colorbar标签

                                # fig.colorbar(im,cax=cbar_ax)
                            ax.axis('off')  # 去除横纵坐标
                            dN += 1
                        plt.figure(num=1)
                        plt.savefig('./log/experiment/Retinex/PasteGD-WH/%d_%d.png' % (i, cc), dpi=200,
                                   bbox_inches='tight',
                                   pad_inches=0)
                        plt.close(fig)  # 关闭当前图形
                        # TCELossa.backward()
            D = D / (Bnum + 1e-8)
            # if TFlag:
            #     (TCELoss / Bnum).backward()
            #     optimizer.step()
            # if not TFlag:
            #     TCELoss = torch.tensor([0]).to(DEVICE)
            record['TCET'] += TCELoss.item()

            CM = running_metric.confuseMT(pr=Tout.squeeze(1).cpu().numpy(), gt=labelT.cpu().numpy())
            lossT = loss_DT2.item() + loss_DS2.item() + loss_DT.item() + loss_DS.item() + TCELoss.item() + SCELoss.item()
            # Score = {'LossT': lossT, 'SCET': SCELoss.item(), 'DGANT': loss_DT.item() + loss_DS.item(), 'TCET': TCELoss.item(),
            #          'D1S':D_out1S.mean().item(),'D1T':D_out1T.mean().item(),'D2S':D_out2S.mean().item(),'D2T':D_out2T.mean().item(),
            #          'EntropyM':EntropyM.mean().item()}
            Score = {'LossT': lossT, 'SCET': SCELoss.item(), 'DGANT': loss_DT.item() + loss_DS.item(),
                     'TCET': TCELoss.item(), 'EntropyM': EntropyM.mean().item()}
            record['LossT'] += lossT

            current_score = running_metric.confuseMS(pr=cd_predSo.cpu().numpy(), gt=labelS.cpu().numpy())
            Score.update(current_score)
            if DA: Score.update(CM)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            # if epoch>0 or opt.load_pretrain:#torch.Size([14, 32, 10, 1])
            if True:
                curcentersT = tool.getCenterSmulti(defeatTDA.detach(), Tout.detach().long(),
                                               centroidsLast=CenterSingel[0, :, :, 0], device=DEVICE)#([2, 32, 14])
                curcentersS = tool.getCenterSmulti(defeatSDA.detach(), labelS.detach().long(),
                                                   centroidsLast=CenterSingel[0, :, :, 0], device=DEVICE)
                curcentersST=torch.cat([curcentersS,curcentersT],dim=-1)#([2, 32, 28])
                Centerori=Center[0,:,:,0]##([32, 10])
                updated_Centerori = Centerori.clone()  # ([32, 10])

                #######unchg Center iter
                distanceunchg = torch.sum((Centerori[:,:unchgN].unsqueeze(2) - curcentersST[0].unsqueeze(1)) ** 2, dim=0)##([5, 28])
                nearest_centroidsunchg = torch.argmin(distanceunchg, dim=0)
                for jj in range( curcentersST[0].shape[1]):  # 遍历所有点
                    point = curcentersST[0][:, jj]
                    nearest_centroid_idx = nearest_centroidsunchg[jj]
                    nearest_centroid = Centerori[:, nearest_centroid_idx]
                    # 更新最近质心的位置
                    updated_Centerori[:, nearest_centroid_idx] += 0.001 * (point - nearest_centroid)

                #######chg Center iter
                distancechg = torch.sum((Centerori[:,unchgN:].unsqueeze(2) - curcentersST[1].unsqueeze(1)) ** 2, dim=0)#([5, 28])
                nearest_centroidschg = torch.argmin(distancechg, dim=0)
                for jj in range( curcentersST[1].shape[1]):  # 遍历所有点
                    point = curcentersST[1][:, jj]
                    nearest_centroid_idx = nearest_centroidschg[jj]
                    nearest_centroid = Centerori[:, unchgN+nearest_centroid_idx]
                    # 更新最近质心的位置
                    updated_Centerori[:, unchgN+nearest_centroid_idx] += 0.001 * (point - nearest_centroid)

                Center=(updated_Centerori.repeat(opt.batch_size, 1, 1)).unsqueeze(-1)#([14, 32, 10, 1])
                CenterSingel = torch.cat([Center[:, :, :unchgN].mean(dim=2),
                                          Center[:, :, unchgN:].mean(dim=2)], dim=-1).unsqueeze(-1)  # ([14, 32, 2, 1])

            ############### Backward Pass ####################
            # update generator weights
            # if i > 10:
            #     break
            if i > 10 and ttest:
                break
        position = int(0.15 * i * opt.batch_size)
        # varentropy=torch.cat(varentropylist,dim=0)
        entropy=torch.cat(entropyThrList['entropymean'],dim=0)
        # varentropyT, indices = torch.sort(varentropy, descending=False)#由小到大
        # varentropyT=varentropyT[position].detach()
        entropyT, indices = torch.sort(entropy, descending=False)  # 由小到大
        entropyT = entropyT[position].detach()

        position = int(0.55 * i * opt.batch_size)
        for ennum in ['entropymean', 'entropyvar', 'entropyunchg', 'entropychg']:
            value = torch.cat(entropyThrList[ennum], dim=0)
            valueThr, indices = torch.sort(value, descending=False)  # 由小到大
            # valueThr=valueThr[position].detach()
            entropyThrOut[ennum + 'Value'] = valueThr[position].detach()
        # print('entropyT',entropyT)
        # print(varentropy.shape,entropy.shape,len(varentropylist))
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        train_scores = running_metric.get_scores()
        IterScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterScore.update(train_scores)
        message = visualizer.print_scores(opt.phase, epoch, IterScore)
        messageT, core_dictT = running_metric.get_scoresT()

        cfg.TRAINLOG.LOGTXT.write(message + '\n')

        exel_out = opt.phase, epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, IterScore['acc'], IterScore[
            'unchgAcc'], \
                   IterScore['chgAcc'], IterScore['recall'], \
                   IterScore['mf1'], IterScore['miou'], IterScore['precision'], \
                   str(IterScore['tn']), str(IterScore['tp']), str(IterScore['fn']), str(IterScore['fp']), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG['wsTrain'].append(exel_out)
        figure_train_metrics = train_metrics.set_figure(metric_dict=figure_train_metrics,
                                                        nochange_acc=IterScore['unchgAcc'],
                                                        change_acc=IterScore['chgAcc'],
                                                        prec=IterScore['precision'], rec=IterScore['recall'],
                                                        f_meas=IterScore['mf1'], total_acc=IterScore['acc'],
                                                        Iou=IterScore['miou'], lossAvg=lossAvg,
                                                        CES_lossAvg=CES_lossAvg,
                                                        DGAN_LossAvg=DGAN_LossAvg, CET_lossAvg=CET_lossAvg)

        ####################################val
        running_metric.clear()
        opt.phase = 'val'
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        val_dataload = val_loader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        tbar = tqdm(range(val_data_len))
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
        print('End of epoch %d / %d \t Time Taken: %d sec \t best acc: %.5f (at epoch: %d) ' %
              (epoch, opt.num_epochs + opt.num_decay_epochs, time.time() - epoch_start_time, best_val_acc, best_epoch))
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

                        # cd_target_pred = net.forward(data_T1_test, data_T2_test)
                        # cd_val_loss, cd_val_pred = cd_model(data['t1_img'].cuda(), data['t2_img'].cuda(), data['label'].cuda())
                        # TCELoss = tool.loss(cd_val_pred[0], label.long())
                        TCELoss = cross_entropy(cd_target_pred[0].detach(), labelT.long())

                        record['TCET'] += TCELoss.item()
                        record['LossT'] = record['TCET']
                        # update metric
                        Score = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                        target_pred = torch.argmax(cd_target_pred[0].detach(), dim=1)
                        current_scoreT = running_metric.confuseMT(pr=target_pred.squeeze(1).cpu().numpy(), gt=labelT.cpu().numpy())

                        # current_scoreT = running_metric.confuseMT(pr=target_pred.cpu().numpy(),
                        #                                          gt=labelT.cpu().numpy())  # 更新
                        Score.update(current_scoreT)
                        valMessageT = visualizer.print_current_scores(opt.phase, epoch, i, t_data_len, Score)
                        tbar.set_description(valMessageT)
                        if i > 10 and ttest:
                            break

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

                    cfg.TRAINLOG.EXCEL_LOG['wsT-' + cfg.TRAINLOG.DATA_NAMES[kk][0]].append(exel_out)
                    figure_T_metrics = T_metrics.set_figure(metric_dict=figure_T_metrics,
                                                            nochange_acc=IterTargetScore['unchgAcc'],
                                                            change_acc=IterTargetScore['chgAcc'],
                                                            prec=IterTargetScore['precision'],
                                                            rec=IterTargetScore['recall'],
                                                            f_meas=IterTargetScore['mf1'],
                                                            total_acc=IterTargetScore['acc'],
                                                            Iou=IterTargetScore['miou'], CES_lossAvg=CET_lossAvg)
                    save_pickle(figure_T_metrics, "./log/%s/%s/fig_T.pkl" % (name, time_now))
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
