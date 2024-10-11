import os
import time
import random
import numpy as np
import torch

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
from util.toolsDCA import adjust_learning_rate,import_config,loss_calc
from util.my_modulesDCA import ICR
from torch.nn.utils import clip_grad
from model.Siamese_diff import FCSiamDiff
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

if __name__ == '__main__':
    gpu_id="0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False,gpu=gpu_id)
    opt.num_epochs=5
    opt.batch_size=28
    opt.use_ce_loss=True
    opt.use_hybrid_loss=False
    opt.num_decay_epochs=0
    opt.dset='CD_DA_building'
    opt.model_type='FCSiamDiff'
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest=False

    saveLast=False
    name = opt.dset + 'CDBase/' + opt.model_type + '_Pretrain'
    opt.LChannel=True
    opt.dataroot=opt.dataroot+'/'+opt.dset
    opt.s = 0
    opt.t=1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'CD_DA_building':
        # cfg.TRAINLOG.DATA_NAMES = ['SYSU_CD', 'LEVIR_CDPatch', 'GZ_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['2LookingPatch256', 'LEVIR_CDPatch', 'WHbuilding256']
        cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatch']
    saveroot=None
    # saveroot = './log/CD_DA_building_DApre/HLCDNetSBN_2/20230328-10_49_SYSU_CD'
    # save_path = saveroot + '/savemodel/_16_acc-0.9012_chgAcc-0.6798_unchgAcc-0.9621.pth'
    opt.load_pretrain = False
    cfg.TRAINLOG.EXCEL_LOGSheet=['wsT-'+cfg.TRAINLOG.DATA_NAMES[0][0],'wsT-'+cfg.TRAINLOG.DATA_NAMES[1][0],'wsTr','wsVal']
    # wsT = [cfg.TRAINLOG.EXCEL_LOG['wsT1'], cfg.TRAINLOG.EXCEL_LOG['wsT2']]
    # wsTr = cfg.TRAINLOG.EXCEL_LOG['wsTr']
    # wsVal = cfg.TRAINLOG.EXCEL_LOG['wsVal']

    print('\n########## Recording File Initialization#################')
    SEED = opt.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg.TRAINLOG.STARTTIME = time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime("%Y%m%d-%H_%M_"+cfg.TRAINLOG.DATA_NAMES[opt.s], time.localtime())
    filename_main = os.path.basename(__file__)
    start_epoch, epoch_iter=MakeRecordFloder(name, time_now, opt,filename_main,opt.load_pretrain,saveroot)
    train_metrics = setFigure()
    val_metrics = setFigure()
    T_metrics = setFigure()
    figure_train_metrics = train_metrics.initialize_figure()
    figure_val_metrics = val_metrics.initialize_figure()
    figure_T_metrics = T_metrics.initialize_figure()
    print('\n########## Load the Source Dataset #################')

    opt.phase = 'train'
    train_loader = CreateDataLoader(opt)
    train_data = train_loader.load_data()
    train_data_len = len(train_data)
    train_size = len(train_loader)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase,cfg.TRAINLOG.DATA_NAMES[opt.s],train_size))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase,cfg.TRAINLOG.DATA_NAMES[opt.s],train_size) + '\n')
    opt.phase = 'val'
    val_loader = CreateDataLoader(opt)
    val_data = val_loader.load_data()
    val_data_len=len(val_data)
    val_size = len(val_loader)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], val_size))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                              (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], val_size) + '\n')

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
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size) + '\n')
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
        figure_train_metrics = load_pickle(saveroot+"/fig_train.pkl")
        figure_val_metrics = load_pickle(saveroot+"/fig_val.pkl")
        start_epoch = len(figure_train_metrics['nochange_acc'])+1
        print('start_epochstart_epochstart_epoch',start_epoch,'end:',opt.num_epochs + opt.num_decay_epochs + 1)
        model_state_dict,bn_domain_map,optimizer_state_dict=tool.load_dackpt(None,OnlyWEIGHTS=False)
        opt.num_epochs=opt.num_epochs+start_epoch
        cfg.DA.BN_DOMAIN_MAP = bn_domain_map

    else:
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}

    print('\n########## Build the Molde #################')
    opt.phase = 'train'
    # net = cfg.TRAINLOG.NETWORK_DICT[opt.model_type]()
    net=FCSiamDiff()
    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(DEVICE)
    print('DEVICE:', DEVICE)
    if model_state_dict is not None:
        model_utils.init_weights(net, model_state_dict, bn_domain_map, False)



    print('\n########## Load the Optimizer #################')

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr,
                                     betas=(0.5, 0.999))
    else:
        raise NotImplemented(opt.optimizer)
    if opt.load_pretrain:
        optimizer.load_state_dict(optimizer_state_dict)
    print('optimizer:', opt.optimizer)
    cfg.TRAINLOG.LOGTXT.write('optimizer: ' + opt.optimizer + '\n')


    visualizer = Visualizer(opt)
    tmp = 1
    running_metric = ConfuseMatrixMeter(n_class=2)
    TRAIN_ACC = np.array([], np.float32)
    VAL_ACC = np.array([], np.float32)
    best_val_acc = 0.0
    best_epoch = 0
    # ws = cfg.TRAINLOG.EXCEL_LOG['Sheet']
    # wsT1 = cfg.TRAINLOG.EXCEL_LOG['wsT1']
    # wsT2 = cfg.TRAINLOG.EXCEL_LOG['wsT2']
    # cfg = import_config('../option/config.py')
    import importlib
    cfgda = importlib.import_module(name='option.DAconfig')
    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        # print('opt.num_epochs', opt.num_epochs)


        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % train_size
        running_metric.clear()
        opt.phase = 'train'
        net.train()
        tbar = tqdm(range(train_data_len))
        iter_train = iter(train_data)
        ce_lossT = 0
        focalLossT = 0
        diceLossT = 0
        LossT = 0
        # net.set_bn_domain(cfg.DA.BN_DOMAIN_MAP[cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]])
        for i in tbar:
            data = next(iter_train)
            epoch_iter += opt.batch_size
            ############## Forward Pass ######################
            optimizer.zero_grad()
            # lr = adjust_learning_rate(optimizer=optimizer, i_iter=(epoch - 1) * train_data_len + i, cfg=cfgda,
            #                           NUM_STEPS_MAX=opt.num_epochs * train_data_len)
            preds1, preds2, feats = net(Variable(data['t1_img']).to(DEVICE), Variable(data['t2_img']).to(DEVICE))

            labelS = Variable(data['label']).to(DEVICE)
            # print('preds1, preds2',preds1.shape, preds2.shape)
            loss_seg = loss_calc([preds1,preds2], labelS, reduction='mean',multi=True)
            source_intra = ICR([preds1,preds2, feats], multi_layer=True)
            # source_intra=torch.Tensor([0]).to(DEVICE)
            loss = loss_seg+source_intra
            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()),
                                      max_norm=35, norm_type=2)
            optimizer.step()

            Score={'CEL':loss_seg.item(),'source_intra':source_intra.item()}
            train_target = data['label'].detach()
            LossT=LossT+loss_seg.item()+loss_seg.item()
            ce_lossT=ce_lossT+loss_seg.item()
            focalLossT=focalLossT+source_intra.item()
            cd_predSo = torch.argmax(preds1.detach(), dim=1)
            current_score = running_metric.update_cm(pr=cd_predSo.cpu().numpy(), gt=train_target.cpu().numpy())
            Score.update(current_score)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            ############### Backward Pass ####################
            # update generator weights
            if i > 10 and ttest:
                break
        lossAvg = LossT / i
        ce_lossAvg = ce_lossT / i
        focalLossAvg = focalLossT / i
        diceLossAvg = diceLossT / i
        train_scores = running_metric.get_scores()
        IterScore = {'Loss':lossAvg,'CE': ce_lossAvg, 'DALoss': 0, 'Dice': diceLossAvg}
        IterScore.update(train_scores)
        message=visualizer.print_scores(opt.phase, epoch, IterScore)
        cfg.TRAINLOG.LOGTXT.write(message+ '\n')

        exel_out = opt.phase, epoch, ce_lossAvg, 0, diceLossAvg, IterScore['acc'], IterScore['unchgAcc'], \
                   IterScore['chgAcc'], IterScore['recall_1'], \
                   IterScore['F1_1'], IterScore['miou'], IterScore['precision_1'],\
                   str(IterScore['tn']), str(IterScore['tp']), str(IterScore['fn']), str(IterScore['fp'])
        # cfg.TRAINLOG.EXCEL_LOG.get_sheet_by_name('wsTr').append(exel_out)
        cfg.TRAINLOG.EXCEL_LOG['wsTr'].append(exel_out)
        figure_train_metrics = train_metrics.set_figure(metric_dict=figure_train_metrics, nochange_acc=IterScore['unchgAcc'],
                                                        change_acc=IterScore['chgAcc'],
                                                        prec=IterScore['precision_1'], rec=IterScore['recall_1'],
                                                        f_meas=train_scores['F1_1'], Loss=lossAvg, ce_loss=ce_lossAvg,
                                                        total_acc=train_scores['acc'],DALoss=0,
                                                        FeatLoss=diceLossAvg, Iou=train_scores['miou'])


        ####################################val
        Tt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        len_target_loader = len(Tt_data)
        running_metric.clear()
        opt.phase = 'val'
        # TTt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        tbar = tqdm(range(len_target_loader))
        iter_val = iter(Tt_data)
        # net.set_bn_domain(cfg.DA.BN_DOMAIN_MAP[cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]])
        ce_lossT = 0
        focalLossT = 0
        diceLossT = 0
        LossT = 0
        val_data_len = len_target_loader
        with torch.no_grad():
            net.eval()
            for i in tbar:
                data_val = next(iter_val)
                # data_T1_val = Variable(data_val['t1_img']).to(DEVICE)
                # data_T2_val = Variable(data_val['t2_img']).to(DEVICE)
                # cd_val_pred = net.forward(data_T1_val,data_T2_val)
                # print('cd_val_pred',cd_val_pred[1].shape)
                label = Variable(data_val['label']).to(DEVICE)
                preds1, preds2, feats = net(Variable(data_val['t1_img']).to(DEVICE), Variable(data_val['t2_img']).to(DEVICE))
                # cd_val_loss, cd_val_pred = cd_model(data['t1_img'].cuda(), data['t2_img'].cuda(), data['label'].cuda())
                val_pred = preds1.detach().max(dim=1)[1].cpu().numpy()
                if opt.use_ce_loss:
                    CELoss = tool.loss(preds1, label.long())
                    # loss = CELoss
                    FocalLoss = 0
                    DiceLoss = 0
                    CELoss = CELoss.item()
                    loss_tr = CELoss + FocalLoss + DiceLoss
                    LossT = LossT + loss_tr
                    ce_lossT = ce_lossT + CELoss
                    focalLoss = focalLossT + FocalLoss
                    diceLossT = diceLossT + DiceLoss

                else:
                    raise NotImplementedError
                # update metric
                Score = {'CEL': CELoss, 'FocalL': FocalLoss, 'DiceL': DiceLoss}
                val_target = data_val['label'].detach()
                # val_pred = torch.argmax(cd_val_pred[0].detach(), dim=1)
                current_score = running_metric.update_cm(pr=val_pred, gt=val_target.cpu().numpy())  # 更新
                Score.update(current_score)
                valMessage = visualizer.print_current_scores(opt.phase, epoch, i, val_data_len, Score)
                tbar.set_description(valMessage)
                if i > 10 and ttest:
                    break
        val_scores = running_metric.get_scores()
        # visualizer.print_scores(opt.phase, epoch, val_scores)

        lossAvg = LossT / i
        ce_lossAvg = ce_lossT / i
        focalLossAvg = focalLossT / i
        diceLossAvg = diceLossT / i
        IterValScore = {'Loss': lossAvg, 'CE': ce_lossAvg, 'DALoss': 0, 'Dice': diceLossAvg}
        IterValScore.update(val_scores)
        message = visualizer.print_scores(opt.phase, epoch, IterValScore)

        cfg.TRAINLOG.LOGTXT.write(message + '\n')

        exel_out = opt.phase, epoch, ce_lossAvg, 0, 0, 0, diceLossAvg, val_scores['acc'], \
                   val_scores['unchgAcc'], val_scores['chgAcc'], val_scores['recall'], \
                   val_scores['mf1'], val_scores['miou'], val_scores['precision_1'], \
                   str(val_scores['tn']), str(val_scores['tp']), str(val_scores['fn']), str(
            val_scores['fp'])

        figure_val_metrics = val_metrics.set_figure(metric_dict=figure_val_metrics, nochange_acc=val_scores['unchgAcc'],
                                                    change_acc=val_scores['chgAcc'],
                                                    prec=val_scores['precision_1'], rec=val_scores['recall'],
                                                    f_meas=val_scores['mf1'], Loss=lossAvg, ce_loss=ce_lossAvg,
                                                    total_acc=val_scores['acc'], FeatLoss=diceLossAvg,
                                                    Iou=val_scores['miou'],DALoss=0)

        val_epoch_acc = val_scores['acc']
        # wsVal.append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.get_sheet_by_name('wsVal').append(exel_out)
        cfg.TRAINLOG.EXCEL_LOG['wsVal'].append(exel_out)

        # VAL_ACC = np.append(VAL_ACC, [val_epoch_acc])
        # np.save(os.path.join(opt.checkpoint_dir, opt.name,  'val_acc.npy'), VAL_ACC)
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch

        if saveLast:
            if epoch == opt.num_epochs:
                save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                           % (
                           name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])
                # tool.save_cd_ckpt(iters=epoch, network=net, optimizer=optimizer, save_str=save_str)
                tool.save_ckpt(network=net, optimizer=optimizer, save_str=save_str)

        else:
            save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                       % (name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])

            # tool.save_cd_ckpt(iters=epoch,network=net,optimizer=optimizer,save_str=save_str)
            tool.save_ckpt(network=net, optimizer=optimizer, save_str=save_str)

        save_pickle(figure_train_metrics, "./log/%s/%s/fig_train.pkl" % (name, time_now))
        save_pickle(figure_val_metrics, "./log/%s/%s/fig_val.pkl" % (name, time_now))

        # end of epoch
        # print(opt.num_epochs,opt.num_decay_epochs)
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec \t best acc: %.5f (at epoch: %d) ' %
            (epoch, opt.num_epochs + opt.num_decay_epochs, time.time() - epoch_start_time, best_val_acc, best_epoch))
        # np.savetxt(cfg.TRAINLOG.ITER_PATH, (epoch + 1, 0), delimiter=',', fmt='%d')
        cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))


                # cfg.TRAINLOG.EXCEL_LOG.get_sheet_by_name(cfg.TRAINLOG.EXCEL_LOGSheet[kk]).append(exel_out)
    print('================ Training Completed (%s) ================\n' % time.strftime("%c"))
    cfg.TRAINLOG.LOGTXT.write('\n================ Training Completed (%s) ================\n' % time.strftime("%c"))
    plotFigureCD(figure_train_metrics, figure_val_metrics, opt.num_epochs + opt.num_decay_epochs, name,opt.model_type, time_now)
    time_end = time.strftime("%Y%m%d-%H_%M", time.localtime())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # if scheduler:
    #     print('Training Start lr:', lr, '  Training Completion lr:', scheduler.get_last_lr())
    print('Training Start Time:', cfg.TRAINLOG.STARTTIME, '  Training Completion Time:', time_end, '  Total Epoch Num:', epoch)
    print('saved path:', './log/{}/{}'.format(name, time_now))


    cfg.TRAINLOG.LOGTXT.write('Training Start Time:'+ cfg.TRAINLOG.STARTTIME+ '  Training Completion Time:'+ time_end+ 'Total Epoch Num:'+ str(epoch) + '\n')





