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
from model.Siamese_diff import FCSiamDiff,FCSiamDiffMaskAttentionCenter
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

if __name__ == '__main__':
    # time.sleep(3000)
    gpu_id = "1"
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
        # cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch', 'GZ_CDPatch']

        # cfg.TRAINLOG.DATA_NAMES = ['S2LookingPatchNoUnchg', 'WHBuildingNoUnchg']

        # cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch', 'GZ_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = [ 'SYSU-CD','S2LookingPatchNoUnchg']

        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatchNoResolution', 'LEVIR_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'WHBuildingNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuilding3', 'GDbuilding']
        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'S2LookingPatchNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['GDbuilding', 'WHBuildingNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = ['GDbuilding2', 'WHBuilding3']
        cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'GDbuilding']

        # cfg.TRAINLOG.DATA_NAMES = [ 'WHBuilding3','GDbuilding']
        # cfg.TRAINLOG.DATA_NAMES = [ 'WHBuildingNoUnchg','S2LookingPatchNoUnchg']
        # cfg.TRAINLOG.DATA_NAMES = [ 'S2LookingPatchNoUnchg','WHBuildingNoUnchg']

    Nepoch = 10
    opt.load_pretrain = False
    if opt.load_pretrain:
        #
        saveroot = './log/CD_DA_buildingCDPretrain2/2-3CenterAttenEntropyMidGAN-itermultiCenter-Paste/20231123-01_20_L-GZ-noPasteSelectEntroNegative/'
        save_path = saveroot + '/savemodel/_11_acc-0.9790_chgAcc-0.5900_unchgAcc-0.9960.pth'
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
        "%Y%m%d-%H_%M_" + cfg.TRAINLOG.DATA_NAMES[opt.s][0] + '-' + cfg.TRAINLOG.DATA_NAMES[opt.t][:2] + '-FCSiameseSource',
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
    net=FCSiamDiff()
    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(DEVICE)

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

    if opt.load_pretrain:
        modelL_state_dict, modelGAN_state_dict, modelGAN2_state_dict, optimizer_state_dict,Centerdict = tool.load_ckptGANCenter(save_path)#,Centerdict
        if modelL_state_dict is not None:
            model_utils.init_weights(net, modelL_state_dict, None, False)
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

    source_label = 0
    target_label = 1
    DA = True
    TagerMetricNum = 0
    zerosLabel = torch.zeros((1, 1, 256, 256)).to(DEVICE)
    cm = plt.cm.get_cmap('jet')
    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):

        opt.phase = 'train'
        #Load Data
        train_data = train_loader.load_data()
        iter_source = iter(train_data)
        train_data_len = len(train_data)
        len_source_loader = train_data_len
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % len(train_loader)
        running_metric.clear()
        net.train()
        tbar = tqdm(range(len_source_loader - 1))
        train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        for i in tbar:
            Sdata = next(iter_source)
            epoch_iter += opt.batch_size
            ############## Forward Pass ######################
            ST1 = Variable(Sdata['t1_img']).to(DEVICE)
            ST2 = Variable(Sdata['t2_img']).to(DEVICE)
            labelS = Variable(Sdata['label']).to(DEVICE)
            cd_predS, defeat3S, defeatSDA = net.forward(ST1, ST2)
            # Generation: source
            optimizer.zero_grad()
            SCELoss = cross_entropy(cd_predS, labelS.long())
            # TCELoss = cross_entropy(cd_predS, labelS.long()).detach()
            TCELoss=torch.tensor([0.0]).to(DEVICE)
            SCELoss.backward()
            record['SCET'] += SCELoss.item()
            # predict
            cd_predSo = torch.argmax(cd_predS.detach(), dim=1)
            #############################Discrimination##################################
            # train with source
            optimizer.step()
            ####Background
            # TCELoss = torch.tensor([0.]).to(DEVICE)
            record['TCET'] += TCELoss.item()
            lossT = TCELoss.item() + SCELoss.item()

            Score = {'LossT': lossT, 'SCET': SCELoss.item(),'TCET': TCELoss.item()}
            record['LossT'] += lossT

            current_score = running_metric.confuseMS(pr=cd_predSo.cpu().numpy(), gt=labelS.cpu().numpy())
            Score.update(current_score)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            if i > 10 and ttest:
                break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        train_scores = running_metric.get_scores()
        IterScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterScore.update(train_scores)
        message = visualizer.print_scores(opt.phase, epoch, IterScore)
        # messageT, core_dictT = running_metric.get_scoresT()

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
            for i in tbar:
                data_val = next(iter_val)
                data_T1_val = Variable(data_val['t1_img']).to(DEVICE)
                data_T2_val = Variable(data_val['t2_img']).to(DEVICE)
                label = Variable(data_val['label']).to(DEVICE)
                cd_val_pred = net.forward(data_T1_val, data_T2_val)

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
                tool.save_ckptGANCenter(network=[net,None,None],Center=[None,None,None], optimizer=optimizer, save_str=save_str)
        else:
            save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                       % (name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])

            # tool.save_cd_ckpt(iters=epoch,network=net,optimizer=optimizer,save_str=save_str)
            # tool.save_ckptGANCenter(network=[net,model_D1,model_D2], optimizer=optimizer, save_str=save_str)
            tool.save_ckptGANCenter(network=[net, None, None], Center=[None,None,None], optimizer=optimizer,
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
                    record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
                    for i in tbar:
                        data_test = next(iter_t)
                        data_T1_test = Variable(data_test['t1_img']).to(DEVICE)
                        data_T2_test = Variable(data_test['t2_img']).to(DEVICE)
                        labelT = Variable(data_test['label']).to(DEVICE)
                        cd_target_pred = net.forward(data_T1_test, data_T2_test)

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
                               IterTargetScore['unchgAcc'], IterTargetScore['chgAcc'], IterTargetScore['recall'], \
                               IterTargetScore['mf1'], IterTargetScore['miou'], IterTargetScore['precision'], \
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
