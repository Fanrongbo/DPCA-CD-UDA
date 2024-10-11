import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
from data.data_loader import CreateDataLoader
from util.train_util import *
from option.train_options import TrainOptions
from util.visualizer import Visualizer
from util.metric_tool2 import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool2 import get_parser_with_args, initialize_weights, save_pickle, load_pickle
from util.drawTool2 import setFigureval, plotFigureTarget, plotFigureDA, MakeRecordFloder, confuseMatrix, plotFigure, \
    setFigureDA
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from util.my_modulesDCA import ICR,CCR
from torch.nn.utils import clip_grad
from model.Siamese_diff import FCSiamDiff
from modelDA.discriminator import FCDiscriminator

def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0
from util.toolsDCA import adjust_learning_rate,import_config,loss_calc


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

    # time.sleep(6000*4)
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
    name = opt.dset + 'CDPretrain2/' + opt.model_type + 'DCA'

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
        cfg.TRAINLOG.DATA_NAMES = ['GDbuilding', 'WHBuildingNoUnchg']

        # cfg.TRAINLOG.DATA_NAMES = [ 'WHBuildingNoUnchg','GDbuilding']
    Nepoch = 10

    opt.load_pretrain = True
    if opt.load_pretrain:
        #
        saveroot = '/data/project_frb/DA/DACDCompare/CDDA2/IntraDA-CD/zmain/log/CD_DA_buildingCDBase/FCSiamDiff/20231215-21_59_G-WH-pretrain/'  #
        save_path = saveroot + '/savemodel/_11_acc-0.9487_chgAcc-0.8302_unchgAcc-0.9537.pth'  # zmain/log/CD_DA_buildingCDBase/FCSiamDiff/20230917-17_19_GZ_CDPatch/
        # opt.num_epochs = 20
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
        "%Y%m%d-%H_%M_" + cfg.TRAINLOG.DATA_NAMES[opt.s][0] + '-' + cfg.TRAINLOG.DATA_NAMES[opt.t][:2] ,
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

    net = FCSiamDiff()
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
    model_D = FCDiscriminator(num_classes=2)

    model_D.to(DEVICE)
    optimizer_D = optim.Adam(model_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    if opt.load_pretrain:
        model_state_dict,bn_domain_map,optimizer_state_dict=tool.load_ckpt(save_path)
        # if model_state_dict is not None:
        model_utils.init_weights(net, model_state_dict, None, False)

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

    entropyT=0
    entropyThrOut={'entropymeanValue': [], 'entropyvarValue': [], 'entropyunchgValue': [], 'entropychgValue': []}

    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        # if (epoch<10 and not opt.load_pretrain) or (opt.load_pretrain and epoch==1) :
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
        model_D.train()
        tbar = tqdm(range(len_target_loader - 1))
        train_data_len = len_target_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        # adjust_learning_rate(optimizer, epoch, opt.num_epochs+1)
        # adjust_learning_rate_D(optimizer_D1, epoch, opt.num_epochs+1)
        # adjust_learning_rate_D(optimizer_D2, epoch, opt.num_epochs+1)


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

            for param in model_D.parameters():
                param.requires_grad = False

            # Generation: source
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            # if epoch<10 or opt.load_pretrain:
            predS1, predS2, featS = net(ST1, ST2)
            predT1, predT2, featT = net(TT1, TT2)
            cd_predTo = torch.argmax(F.softmax(predT1, dim=1), dim=1).unsqueeze(1)  #
            loss_seg = loss_calc([predS1, predS2], labelS, multi=True)
            loss_pseudo = loss_calc([predT1, predT2], cd_predTo, multi=True)
            source_intra = ICR([predS1, predS2, featS],
                               multi_layer=True)
            domain_cross = CCR([predS1, predS2, featS],
                               [predT1, predT2, featT],
                               multi_layer=True)
            loss = loss_seg + loss_pseudo + (source_intra + domain_cross)
            record['SCET'] = record['SCET'] + loss_seg.item()
            record['LossT'] = record['LossT'] + loss.item()
            record['DGANT'] = record['DGANT'] + (source_intra + domain_cross).item()
            optimizer.zero_grad()
            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()),
                                      max_norm=35, norm_type=2)
            optimizer.step()

            cd_predSo = torch.argmax(F.softmax(predS1, dim=1), dim=1)  #
            softmaxpreT = F.softmax(predT1, dim=1)
            Tout = torch.argmax(softmaxpreT.detach(), dim=1)
            optimizer.step()
            # optimizer_D.step()

            #
            # if not TFlag:
            #     TCELoss = torch.tensor([0]).to(DEVICE)
            # record['LossT'] += TCELoss.item()

            # CM = running_metric.confuseMS(pr=Tout.squeeze(1).cpu().numpy(), presoft=None, gt=labelT.cpu().numpy())
            CM = running_metric.confuseMT(pr=Tout.squeeze(1).cpu().numpy(), gt=labelT.cpu().numpy())

            Score = {'CEL': loss_seg.item(), 'loss_pseudo': loss_pseudo.item(),
                     'DA': (source_intra + domain_cross).item()}
            # Score = {'LossT': lossT, 'SCET': SCELoss.item(), 'DGANT': loss_DT.item() + loss_DS.item(), 'TCET': TCELoss.item(),
            #          'D1S':D_out1S.mean().item(),'D1T':D_out1T.mean().item(),'D2S':D_out2S.mean().item(),'D2T':D_out2T.mean().item(),
            #          'EntropyM':EntropyM.mean().item()}

            current_score = running_metric.confuseMS(pr=cd_predSo.cpu().numpy(), gt=labelS.cpu().numpy())
            Score.update(current_score)
            if DA: Score.update(CM)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            # if epoch>0 or opt.load_pretrain:#torch.Size([14, 32, 10, 1])


            ############### Backward Pass ####################
            # update generator weights
            # if i > 10:
            #     break
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
            model_D.eval()
            for i in tbar:
                data_val = next(iter_val)
                data_T1_val = Variable(data_val['t1_img']).to(DEVICE)
                data_T2_val = Variable(data_val['t2_img']).to(DEVICE)
                label = Variable(data_val['label']).to(DEVICE)
                preds1, preds2, feats = net(Variable(data_val['t1_img']).to(DEVICE),
                                            Variable(data_val['t2_img']).to(DEVICE))
                val_pred = preds1.detach().max(dim=1)[1].cpu().numpy()

                TCELoss = tool.loss(preds1, label.long())
                record['SCET'] += TCELoss.item()
                record['LossT'] = record['SCET']
                # update metric
                Score = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                val_data = data_val['label'].detach()
                # val_pred = torch.argmax(F.softmax(cd_val_pred[0], dim=1), dim=1)

                current_score = running_metric.confuseMS(pr=val_pred, gt=val_data.cpu().numpy())  # 更新
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
                tool.save_ckptGANCenter(network=[net,model_D,model_D],Center=[None,None,None], optimizer=optimizer, save_str=save_str)


        else:
            save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                       % (name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])

            # tool.save_cd_ckpt(iters=epoch,network=net,optimizer=optimizer,save_str=save_str)
            # tool.save_ckptGANCenter(network=[net,model_D1,model_D2], optimizer=optimizer, save_str=save_str)
            tool.save_ckptGANCenter(network=[net, model_D, model_D], Center=[None,None,None], optimizer=optimizer,
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
                    model_D.eval()
                    record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
                    for i in tbar:
                        data_test = next(iter_t)
                        # data_T1_test = Variable(data_test['t1_img']).to(DEVICE)
                        # data_T2_test = Variable(data_test['t2_img']).to(DEVICE)
                        labelT = Variable(data_test['label']).to(DEVICE)
                        preds1, preds2, feats = net(Variable(data_test['t1_img']).to(DEVICE),
                                                    Variable(data_test['t2_img']).to(DEVICE))
                        # cd_target_pred = net.forward(data_T1_test, data_T2_test)
                        # cd_val_loss, cd_val_pred = cd_model(data['t1_img'].cuda(), data['t2_img'].cuda(), data['label'].cuda())
                        # TCELoss = tool.loss(cd_val_pred[0], label.long())
                        # TCELoss = cross_entropy(cd_target_pred[0].detach(), labelT.long())
                        target_pred = preds1.detach().max(dim=1)[1].cpu().numpy()

                        TCELoss = tool.loss(preds1, label.long())
                        record['TCET'] += TCELoss.item()
                        record['LossT'] = record['TCET']
                        # update metric
                        Score = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                        # target_pred = torch.argmax(cd_target_pred, dim=1)
                        current_scoreT = running_metric.confuseMT(pr=target_pred, gt=labelT.cpu().numpy())

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
