
import random

from data.data_loader import CreateDataLoader
# from model.cd_model import *
from util.train_util import *

from option.train_options import TrainOptions
from util.visualizer import Visualizer
from util.metric_tool import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool import get_parser_with_args,initialize_weights,save_pickle,load_pickle
from util.drawTool import setFigure,add_weight_decay,plotFigureCD,MakeRecordFloder,confuseMatrix,plotFigure,setFigureDA
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils


def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
def adjust_learning_rate(optimizer, i_iter):
    LEARNING_RATE = 2.5e-4
    lr = lr_poly(LEARNING_RATE, i_iter, 250000, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(1e-4, i_iter, 250000, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
if __name__ == '__main__':
    gpu_id="0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False,gpu=gpu_id)
    opt.num_epochs=20
    opt.batch_size=20
    opt.use_ce_loss=True
    opt.use_hybrid_loss=False
    opt.num_decay_epochs=0
    opt.dset='CD_DA_building'
    opt.model_type='FCSiamDiff'
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest=False

    saveLast=True
    name = opt.dset + 'CDDA/' + opt.model_type + 'stage1'
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
    # saveroot=None//zmain/log/CD_DA_buildingCDBase/FCSiamDiff/20230724-11_59_WHbuilding256/savemodel/_31_acc-0.9034_chgAcc-0.8867_unchgAcc-0.9037.pth
    saveroot = '/data/project_frb/DA/DACDCompare/EasytoHardAdvent/zmain/log/CD_DA_buildingCDDA/FCSiamDiffAdvEntStage1/20230924-17_49_GZ_CDPatch/'#
    save_path = saveroot + '/savemodel/_2_acc-0.9512_chgAcc-0.0092_unchgAcc-0.9973.pth'#/data/project_frb/DA/DACDCompare/EasytoHardAdvent/zmain/log/CD_DA_buildingCDDA/FCSiamDiffAdvEntStage1/20230924-17_49_GZ_CDPatch/
    opt.load_pretrain = True
    cfg.TRAINLOG.EXCEL_LOGSheet=['wsT-'+cfg.TRAINLOG.DATA_NAMES[0][0],'wsT-'+cfg.TRAINLOG.DATA_NAMES[1][0],
                                 'wsTr','wsVal']
    # wsT = [cfg.TRAINLOG.EXCEL_LOG['wsT1'], cfg.TRAINLOG.EXCEL_LOG['wsT2']]
    # wsTr = cfg.TRAINLOG.EXCEL_LOG['wsTr']
    # wsVal = cfg.TRAINLOG.EXCEL_LOG['wsVal']
    opt.hardspilt='/data/project_frb/DA/DACDCompare/MADACD/zmain/output/G-L/stage1_cac_list_0.05.txt'
    print('\n########## Recording File Initialization#################')
    SEED = opt.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg.TRAINLOG.STARTTIME = time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime("%Y%m%d-%H_%M_"+cfg.TRAINLOG.DATA_NAMES[opt.s]+'', time.localtime())
    kernelsize=32
    kk=1
    filename_main = os.path.basename(__file__)
    start_epoch, epoch_iter=MakeRecordFloder(name, time_now, opt,filename_main,opt.load_pretrain,saveroot)
    train_metrics = setFigureDA()
    val_metrics = setFigureDA()
    figure_train_metrics = train_metrics.initialize_figure()
    figure_val_metrics = val_metrics.initialize_figure()
    # figure_T_metrics = T_metrics.initialize_figure()
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
            opt.phase = 'targetSelect'
            t_loader = CreateDataLoader(opt)
            t_loaderDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            t_data = t_loader.load_data()
            t_data_len = len(t_data)
            t_size = len(t_loader)
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size) + '\n')
    t_loaderDictVal = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
            opt.t = i
            opt.phase = 'target'
            t_loader = CreateDataLoader(opt)
            t_loaderDictVal.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            t_data = t_loader.load_data()
            t_data_len = len(t_data)
            t_size = len(t_loader)
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size) + '\n')
    # print(t_loaderDict)
    tool = CDModelutil(opt)
    opt.phase = 'train'
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

    print('\n########## Load the Optimizer #################')
    LEARNING_RATE = 2.5e-4

    visualizer = Visualizer(opt)
    tmp = 1
    running_metric = ConfuseMatrixMeter(n_class=2)
    TRAIN_ACC = np.array([], np.float32)
    VAL_ACC = np.array([], np.float32)
    best_val_acc = 0.0
    best_epoch = 0
    ###################################################
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr,
                                     betas=(0.5, 0.999))
    else:
        raise NotImplemented(opt.optimizer)
    DA=True
    #################ADsegInit#################################################
    # model_D2 = FCDiscriminator(num_classes=2)
    # model_D2.to(DEVICE)

    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        # print('opt.num_epochs', opt.num_epochs)
        # d_aux.train()
        Tt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        len_target_loader = len(Tt_data)
        # tbar = tqdm(range(t_data_len))
        iter_target = iter(Tt_data)
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % train_size
        running_metric.clear()
        opt.phase = 'train'
        train_data = train_loader.load_data()
        iter_source = iter(train_data)
        FeatLossT=0
        ce_lossT = 0
        focalLossT = 0
        diceLossT = 0
        LossT = 0
        len_source_loader = train_data_len
        net.train()
        print('len_source_loader',len_source_loader,len_target_loader)
        # net.set_bn_domain(cfg.DA.BN_DOMAIN_MAP[cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]])
        if len_source_loader > len_target_loader:
            tbar = tqdm(range(len_source_loader - 1))
            train_data_len = len_source_loader
        else:
            tbar = tqdm(range(len_target_loader - 1))
            train_data_len = len_target_loader

        for i in tbar:
            try:
                Sdata = next(iter_source)
            except:
                iter_source = iter(train_data)
            try:
                Tdata = next(iter_target)
            except:
                iter_target = iter(t_loader)
            epoch_iter += opt.batch_size
            optimizer.zero_grad()
            net.train()
            #############################Generation##################################


            TT1 = Variable(Tdata['t1_img']).to(DEVICE)
            TT2 = Variable(Tdata['t2_img']).to(DEVICE)
            labelT = Variable(Tdata['label']).to(DEVICE)
            cd_predT, defeat3T, defeatTDA = net(TT1, TT2)
            CELossAC = tool.loss(cd_predT, labelT.long())
            # CELossAC.backward()

            ST1 = Variable(Sdata['t1_img']).to(DEVICE)
            ST2 = Variable(Sdata['t2_img']).to(DEVICE)
            labelS = Variable(Sdata['label']).to(DEVICE)
            cd_predS, defeat3S, defeatSDA = net(ST1, ST2)
            CELoss = tool.loss(cd_predS, labelS.long())  # CE
            # loss = loss_seg1
            (CELoss+CELossAC).backward()
            # net.eval()
            optimizer.step()

            cd_predSo = torch.argmax(cd_predS.detach(), dim=1)
            softmaxpreT = F.softmax(cd_predT, dim=1)
            Tout = torch.argmax(softmaxpreT.detach(), dim=1)

            # optimizer_d_aux.step()
            CM = running_metric.confuseM(pr=Tout.squeeze(1).cpu().numpy(), presoft=None, gt=labelT.cpu().numpy())
            Score={'CEL':CELoss,'DALoss':CELossAC.item()}

            current_score = running_metric.update_cm(pr=cd_predSo.cpu().numpy(), gt=labelS.cpu().numpy())
            Score.update(current_score)
            if DA: Score.update(CM)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            ############### Backward Pass ####################
            # update generator weights
            # if i > 10:
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
        messageT,core_dictT = running_metric.get_scoresT()


        cfg.TRAINLOG.LOGTXT.write(message+ '\n')

        exel_out = opt.phase, epoch, ce_lossAvg, 0, 0, 0, diceLossAvg, IterScore['acc'], IterScore['unchgAcc'], \
                   IterScore['chgAcc'], IterScore['recall'], \
                   IterScore['mf1'], IterScore['miou'], IterScore['precision'],\
                   str(IterScore['tn']), str(IterScore['tp']), str(IterScore['fn']), str(IterScore['fp']),\
                   core_dictT['accT'],core_dictT['unchgT'],core_dictT['chgT'],core_dictT['mF1T']
        # cfg.TRAINLOG.EXCEL_LOG.get_sheet_by_name('wsTr').append(exel_out)
        cfg.TRAINLOG.EXCEL_LOG['wsTr'].append(exel_out)
        figure_train_metrics = train_metrics.set_figure(metric_dict=figure_train_metrics, nochange_acc=IterScore['unchgAcc'],
                                                        change_acc=IterScore['chgAcc'],
                                                        prec=IterScore['precision'], rec=IterScore['recall'],
                                                        f_meas=train_scores['mf1'], Loss=lossAvg, ce_loss=ce_lossAvg,
                                                        total_acc=train_scores['acc'],DAlowLoss=0,DAEntropy=0,DATLoss=0,
                                                        FeatLoss=diceLossAvg, Iou=train_scores['miou'])
        ####################################val

        Tt_loader = t_loaderDictVal[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        len_target_loader = len(Tt_data)
        # # tbar = tqdm(range(t_data_len))
        # iter_target = iter(Tt_data)

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
                data_T1_val=Variable(data_val['t1_img']).to(DEVICE)
                data_T2_val=Variable(data_val['t2_img']).to(DEVICE)
                cd_val_pred = net.forward(data_T1_val,data_T2_val)
                # print('cd_val_pred',cd_val_pred[1].shape)
                label=Variable(data_val['label']).to(DEVICE)

                # cd_val_loss, cd_val_pred = cd_model(data['t1_img'].cuda(), data['t2_img'].cuda(), data['label'].cuda())

                if opt.use_ce_loss:
                    CELoss = tool.loss(cd_val_pred[0], label.long())
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
                val_pred = torch.argmax(cd_val_pred[0].detach(), dim=1)
                current_score = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())#更新
                Score.update(current_score)
                valMessage=visualizer.print_current_scores(opt.phase,epoch,i,val_data_len,Score)
                tbar.set_description(valMessage)
                # if i > 10 and ttest:
                #     break
        val_scores = running_metric.get_scores()
        # visualizer.print_scores(opt.phase, epoch, val_scores)

        lossAvg = LossT / i
        ce_lossAvg = ce_lossT / i
        focalLossAvg = focalLossT / i
        diceLossAvg = diceLossT / i
        IterValScore = {'Loss': lossAvg, 'CE': ce_lossAvg, 'DALoss': 0, 'Dice': diceLossAvg}
        IterValScore.update(val_scores)
        message=visualizer.print_scores(opt.phase, epoch, IterValScore)

        cfg.TRAINLOG.LOGTXT.write(message + '\n')

        exel_out = opt.phase, epoch, ce_lossAvg, 0, 0,0,diceLossAvg, val_scores['acc'], \
                   val_scores['unchgAcc'],val_scores['chgAcc'], val_scores['recall'], \
                   val_scores['mf1'], val_scores['miou'], val_scores['precision_1'], \
                   str(val_scores['tn']), str(val_scores['tp']), str(val_scores['fn']), str(
            val_scores['fp'])

        figure_val_metrics = val_metrics.set_figure(metric_dict=figure_val_metrics, nochange_acc=val_scores['unchgAcc'],
                                                        change_acc=val_scores['chgAcc'],
                                                        prec=val_scores['precision_1'], rec=val_scores['recall'],
                                                        f_meas=val_scores['mf1'], Loss=lossAvg, ce_loss=ce_lossAvg,
                                                        total_acc=val_scores['acc'],
                                                        DAlowLoss=0,DAEntropy=0,DATLoss=0, FeatLoss=diceLossAvg, Iou=val_scores['miou'])

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





