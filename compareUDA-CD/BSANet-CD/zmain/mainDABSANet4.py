
import random


from data.data_loader import CreateDataLoader
# from model.cd_model import *
from util.train_util import *
from util.metric_tool2 import ConfuseMatrixMeter

from option.train_options import TrainOptions
from util.visualizer import Visualizer
# from util.metric_tool import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool import get_parser_with_args,initialize_weights,save_pickle,load_pickle
from util.drawTool import setFigure,add_weight_decay,plotFigureCD,MakeRecordFloder,confuseMatrix,plotFigure,setFigureDA
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils
from util.metrics_DAcva import CORAL,MMD_loss,SelecFeat,MMD_lossclass3,CVACD
from model.Siamese_diff import genpatch,PatchLoss,centerDist,genMaskPatchNew
from modelDA.mmd import MMD
from modelDA.discriminator import layoutDiscriminator,Discriminator
from modelDA import BSANet
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--pad', type=str, default='zero', help='pad type of networks')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type of networks')
    parser.add_argument('--act', type = str, default = 'relu', help = 'activation type of networks')

    return parser.parse_args()
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
    # time.sleep(60000)
    gpu_id="1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False,gpu=gpu_id)
    opt.num_epochs=30
    opt.batch_size=10
    opt.use_ce_loss=True
    opt.use_hybrid_loss=False
    opt.num_decay_epochs=0

    opt.dset='CD_DA_building'
    opt.model_type='BSANet'
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest=False
    saveLast=True
    name = opt.dset + 'CDDA/' + opt.model_type + 'DoubleGAN'
    opt.LChannel=True
    opt.dataroot=opt.dataroot+'/'+opt.dset
    opt.s = 0
    opt.t=1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'CD_DA_building':
        # cfg.TRAINLOG.DATA_NAMES = ['SYSU_CD', 'LEVIR_CDPatch', 'GZ_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['2LookingPatch256', 'LEVIR_CDPatch', 'WHbuilding256']
        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatchNoResolution']
        # cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch', 'GZ_CDPatch']
        # cfg.TRAINLOG.DATA_NAMES = ['GZ_CDPatch', 'LEVIR_CDPatch']

        # cfg.TRAINLOG.DATA_NAMES = ['WHBuildingNoUnchg', 'GDbuilding']
        cfg.TRAINLOG.DATA_NAMES = ['GDbuilding', 'WHBuildingNoUnchg']


    # saveroot=None//zmain/log/CD_DA_buildingCDBase/FCSiamDiff/20230724-11_59_WHbuilding256/savemodel/_31_acc-0.9034_chgAcc-0.8867_unchgAcc-0.9037.pth
    # saveroot = '/data/project_frb/DA/DACDCompare/CDDA/BSANet-CD/zmain/log/CD_DA_buildingCDBase/BSANet/20231212-21_11_GZ_CDPatch/'#
    # save_path = saveroot + '/savemodel/_10_acc-0.8979_chgAcc-0.2529_unchgAcc-0.9876.pth'#zmain/log/CD_DA_buildingCDBase/BSANet/20230923-14_27_GZ_CDPatch/savemodel/
    saveroot = '/data/project_frb/DA/DACDCompare/CDDA/BSANet-CD/zmain/log/CD_DA_buildingCDBase/BSANet/20231213-17_19_GDbuilding/'  #
    save_path = saveroot + '/savemodel/_11_acc-0.9886_chgAcc-0.8335_unchgAcc-0.9952.pth'  # zmain/log/CD_DA_buildingCDBase/BSANet/20230923-14_27_GZ_CDPatch/savemodel/

    opt.load_pretrain = False
    cfg.TRAINLOG.EXCEL_LOGSheet=['wsT-'+cfg.TRAINLOG.DATA_NAMES[0][0],'wsT-'+cfg.TRAINLOG.DATA_NAMES[1][0],
                                 'wsTr','wsVal']
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
    time_now = time.strftime("%Y%m%d-%H_%M_"+cfg.TRAINLOG.DATA_NAMES[opt.s]+'_k=1-s=32', time.localtime())
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
        model_state_dict,bn_domain_map,optimizer_state_dict=tool.load_ckpt(save_path)
        cfg.DA.BN_DOMAIN_MAP = bn_domain_map
    else:
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}

    print('\n########## Build the Molde #################')
    opt.phase = 'train'
    # net = cfg.TRAINLOG.NETWORK_DICT[opt.model_type]()
    args = get_arguments()
    net = BSANet.Generator(args, 2)

    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(DEVICE)
    print('DEVICE:', DEVICE)
    if model_state_dict is not None:
        model_utils.init_weights(net, model_state_dict, bn_domain_map, False)

    print('\n########## Load the Optimizer #################')
    LEARNING_RATE = 2.5e-4
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE,
                                    momentum=0.9,
                                    weight_decay=0.0005)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr,
                                     betas=(0.5, 0.999))
    else:
        raise NotImplemented(opt.optimizer)
    if opt.load_pretrain:
        optimizer.load_state_dict(optimizer_state_dict)
    print('optimizer:', opt.optimizer)
    cfg.TRAINLOG.LOGTXT.write('optimizer: ' + opt.optimizer + '\n')
    import cv2

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
    DA=True
    CVACDProcess=CVACD().to(DEVICE)
    genmaskPatch=genMaskPatchNew(orisize=256,device=DEVICE,kernel=kernelsize).to(DEVICE)#genMaskPatch genpatch
    genPatch2=genpatch(orisize=256,device=DEVICE,kernel=kernelsize).to(DEVICE)#genMaskPatch genpatch

    patchLoss=PatchLoss()
    selectF = SelecFeat()
    mmd = MMD(num_layers=1, kernel_num=[3],
              kernel_mul=[2], joint=False, device=DEVICE)
    centerDA=centerDist(device=DEVICE).to(DEVICE)
    mse=torch.nn.MSELoss(reduction='mean')

    #################ADsegInit#################################################
    model_D1 = layoutDiscriminator(num_classes=2).to(DEVICE)
    model_D2 = Discriminator(num_classes=64).to(DEVICE)


    # model_D2.to(DEVICE)
    LEARNING_RATE_D = 1e-4
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    bce_loss = torch.nn.BCEWithLogitsLoss()

    gan = 'Vanilla'
    if gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    interp = nn.Upsample(size=(256, 256), mode='bilinear')
    source_label = 0
    target_label = 1
    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        # print('opt.num_epochs', opt.num_epochs)
        model_D1.train()
        model_D2.train()

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
        model_D1.train()
        model_D2.train()
        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        loss_adv_target_value3 = 0
        loss_D_value3 = 0

        loss_adv_target_value4 = 0
        loss_D_value4 = 0

        loss_adv_target_value5 = 0
        loss_D_value5 = 0
        adjust_learning_rate_D(optimizer_D1, epoch)
        adjust_learning_rate_D(optimizer_D2, epoch)
        adjust_learning_rate(optimizer, epoch)

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

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer.zero_grad()

            #############################Generation##################################

            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            ST1 = Variable(Sdata['t1_img']).to(DEVICE)
            ST2 = Variable(Sdata['t2_img']).to(DEVICE)
            labelS = Variable(Sdata['label']).to(DEVICE)
            pred1, merge_src, concat_src, final_src, U_middle_src, W_middle_src = net(ST1, ST2)
            CELoss = tool.loss(pred1, labelS.long())  # CE
            # loss = loss_seg1
            CELoss.backward(retain_graph=True)

            TT1 = Variable(Tdata['t1_img']).to(DEVICE)
            TT2 = Variable(Tdata['t2_img']).to(DEVICE)
            labelT = Variable(Tdata['label']).to(DEVICE)
            pred_target1, merge_tar, concat_tar, final_tar, U_middle_tar, W_middle_tar = net(TT1, TT2)
            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(final_tar)

            loss_GT1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).to(DEVICE))
            loss_GT2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).to(DEVICE))
            LAMBDA_ADV_TARGET1 = 0.5
            LAMBDA_ADV_TARGET2 = 0.5
            loss_GT = LAMBDA_ADV_TARGET1 * loss_GT1 + LAMBDA_ADV_TARGET2 * loss_GT2
            loss_GT.backward(retain_graph=True)
            optimizer.step()

            # loss=loss_adv_target1
            #############################Discrimination##################################
            # train with source
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            pred1 = pred1.detach()
            final_src = final_src.detach()

            D_out1 = model_D1(F.softmax(pred1,dim=1))
            D_out2 = model_D2(final_src)
            # print('final_src',final_src.shape)
            loss_DS = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).to(DEVICE))
            loss_DS2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).to(DEVICE))
            # loss_DS.backward()
            loss_DS.backward(retain_graph=True)
            loss_DS2.backward(retain_graph=True)
            # train with target
            pred_target1 = pred_target1.detach()
            final_tar = final_tar.detach()

            D_out1 = model_D1(F.softmax(pred_target1,dim=1))
            D_out2 = model_D2(final_tar)
            loss_DT = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).to(DEVICE))
            loss_DT2 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).to(DEVICE))
            loss_DT.backward(retain_graph=True)
            loss_DT2.backward()
            optimizer_D1.step()
            optimizer_D2.step()

            cd_predS= F.softmax(pred1, dim=1)
            cd_predSo = torch.argmax(cd_predS.detach(), dim=1)
            softmaxpreT = F.softmax(pred_target1, dim=1)
            Tout = torch.argmax(softmaxpreT.detach(), dim=1)

            CM = running_metric.confuseMT(pr=Tout.squeeze(1).cpu().numpy(), gt=labelT.cpu().numpy())
            Score={'CEL':CELoss,'DALoss':loss_DT.item()+loss_DS.item(),'DiceL':loss_GT.item()}

            current_score = running_metric.confuseMS(pr=cd_predSo.cpu().numpy(), gt=labelS.cpu().numpy())
            Score.update(current_score)
            if DA: Score.update(CM)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            ############### Backward Pass ####################
            # update generator weights
            if i>300:
                break
            # if i > 300 and ttest:
            #     break
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
                   core_dictT['accT'],core_dictT['unchgT'],core_dictT['chgT'],core_dictT['mF1T'],core_dictT['miouT']
        # cfg.TRAINLOG.EXCEL_LOG.get_sheet_by_name('wsTr').append(exel_out)
        cfg.TRAINLOG.EXCEL_LOG['wsTr'].append(exel_out)
        figure_train_metrics = train_metrics.set_figure(metric_dict=figure_train_metrics, nochange_acc=IterScore['unchgAcc'],
                                                        change_acc=IterScore['chgAcc'],
                                                        prec=IterScore['precision'], rec=IterScore['recall'],
                                                        f_meas=train_scores['mf1'], Loss=lossAvg, ce_loss=ce_lossAvg,
                                                        total_acc=train_scores['acc'],DAlowLoss=0,DAEntropy=0,DATLoss=0,
                                                        FeatLoss=diceLossAvg, Iou=train_scores['miou'])

        ####################################val

        running_metric.clear()
        opt.phase = 'val'
        tbar = tqdm(range(val_data_len))
        iter_val = iter(val_data)
        # net.set_bn_domain(cfg.DA.BN_DOMAIN_MAP[cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]])
        ce_lossT = 0
        focalLossT = 0
        diceLossT = 0
        LossT = 0
        with torch.no_grad():
            net.eval()
            for i in tbar:
                data_val = next(iter_val)
                data_T1_val = Variable(data_val['t1_img']).to(DEVICE)
                data_T2_val = Variable(data_val['t2_img']).to(DEVICE)
                cd_val_pred = net.forward(data_T1_val, data_T2_val)
                # print('cd_val_pred',cd_val_pred[1].shape)
                label = Variable(data_val['label']).to(DEVICE)
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
                elif opt.use_hybrid_loss:
                    FocalLoss, DiceLoss = tool.loss(cd_val_pred[0], label.long())
                    # loss = FocalLoss + DiceLoss
                    CELoss = 0
                    FocalLoss = FocalLoss.item()
                    DiceLoss = DiceLoss.item()
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
                current_score = running_metric.confuseMS(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())  # 更新
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
        message=visualizer.print_scores(opt.phase, epoch, IterValScore)

        cfg.TRAINLOG.LOGTXT.write(message + '\n')

        exel_out = opt.phase, epoch, ce_lossAvg, 0, 0,0,diceLossAvg, val_scores['acc'], \
                   val_scores['unchgAcc'],val_scores['chgAcc'], val_scores['recall'], \
                   val_scores['mf1'], val_scores['miou'], val_scores['precision'], \
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

        # save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
        #            % (name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])
        # tool.save_ckpt(network=net, optimizer=optimizer, save_str=save_str)
        # print('save',time.time()-epoch_start_time)
        # iter_save_str='last_epoch.pth'
        # print(name, time_now, name,)
        # print(save_str)
        # torch.save(net.state_dict(), save_str)f

        # tool.save_ckptDA(iters=epoch,network=net,bn_domain_map=cfg.DA.BN_DOMAIN_MAP,optimizer=optimizer,save_str=save_str)

        save_pickle(figure_train_metrics, "./log/%s/%s/fig_train.pkl" % (name, time_now))
        save_pickle(figure_val_metrics, "./log/%s/%s/fig_val.pkl" % (name, time_now))
        cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))

        # cfg.TRAINLOG.LOGTXT.write('\n================ Target Test (%s) ================\n' % time.strftime("%c"))
        if epoch % 1 == 0:
            print('================epoch:%d Target Test (%s) ================\n' % (epoch, time.strftime("%c")))
            cfg.TRAINLOG.LOGTXT.write(
                '\n================epoch:%d Target Test (%s) ================\n' % (epoch, time.strftime("%c")))
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])

                t_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                t_data = t_loader.load_data()
                t_data_len = len(t_data)
                tbar = tqdm(range(t_data_len))
                iter_t = iter(t_data)
                running_metric.clear()
                opt.phase = 'val'
                with torch.no_grad():
                    net.eval()
                    ce_lossT = 0
                    focalLossT = 0
                    diceLossT = 0
                    LossT = 0
                    for i in tbar:
                        # for i, data in enumerate(val_data):
                        #     data=iter_val.next()
                        data_test = next(iter_t)
                        data_T1_val = Variable(data_test['t1_img']).to(DEVICE)
                        data_T2_val = Variable(data_test['t2_img']).to(DEVICE)
                        cd_val_pred = net.forward(data_T1_val, data_T2_val)

                        label = Variable(data_test['label']).to(DEVICE)

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
                        elif opt.use_hybrid_loss:
                            FocalLoss, DiceLoss = tool.loss(cd_val_pred[0], label.long())
                            # loss = FocalLoss + DiceLoss
                            CELoss = 0
                            FocalLoss = FocalLoss.item()
                            DiceLoss = DiceLoss.item()
                            loss_tr = CELoss + FocalLoss + DiceLoss
                            LossT = LossT + loss_tr
                            ce_lossT = ce_lossT + CELoss
                            focalLoss = focalLossT + FocalLoss
                            diceLossT = diceLossT + DiceLoss

                        else:
                            raise NotImplementedError
                        # update metric
                        Score = {'CEL': CELoss, 'DALoss': 0, 'DiceL': DiceLoss}
                        val_target = data_test['label'].detach()
                        val_pred = torch.argmax(cd_val_pred[0].detach(), dim=1)
                        # current_score = running_metric.update_cm(pr=val_pred.cpu().numpy(),
                        #                                          gt=val_target.cpu().numpy())  # 更新
                        current_scoreT = running_metric.confuseMT(pr=val_pred.squeeze(1).cpu().numpy(),
                                                                  gt=val_target.cpu().numpy())

                        Score.update(current_scoreT)
                        valMessage = visualizer.print_current_scores('target', epoch, i, t_data_len, Score)
                        tbar.set_description(valMessage)
                        if i > 10 and ttest:
                            break
                target_scores = running_metric.get_scores()
                # visualizer.print_scores('target out', epoch, val_scores)

                lossAvg = LossT / i
                ce_lossAvg = ce_lossT / i
                focalLossAvg = focalLossT / i
                diceLossAvg = diceLossT / i
                IterValScore = {'Loss': lossAvg, 'CE': ce_lossAvg, 'DALoss': 0, 'Dice': diceLossAvg}
                IterValScore.update(target_scores)
                message = visualizer.print_scores('target out', epoch, IterValScore)
                cfg.TRAINLOG.LOGTXT.write('\n================ [%s] Target Test (%s) ================\n' % (
                    cfg.TRAINLOG.DATA_NAMES[kk], time.strftime("%c")))

                cfg.TRAINLOG.LOGTXT.write(message + '\n')

                # exel_out = opt.phase, epoch, ce_lossAvg, 0, 0, 0, 0, val_scores['acc'], \
                #            val_scores['unchgAcc'], val_scores['chgAcc'], val_scores['recall'], \
                #            val_scores['mf1'], val_scores['miou'], val_scores['precision'], \
                #            str(val_scores['tn']), str(val_scores['tp']), str(val_scores['fn']), str(
                #     val_scores['fp'])

                exel_out = 'T-' + cfg.TRAINLOG.DATA_NAMES[kk][0], epoch, ce_lossAvg, 0, 0, 0, 0, target_scores['acc'], \
                           target_scores['unchgAcc'], target_scores['chgAcc'], target_scores['recall_1'], \
                           target_scores['F1_1'], target_scores['miou'], target_scores['precision_1'], \
                           str(target_scores['tn']), str(target_scores['tp']), str(target_scores['fn']), str(
                    target_scores['fp'])
                # wsT[kk].append(exel_out)
                # 'wsT-G'
                cfg.TRAINLOG.EXCEL_LOG['wsT-' + cfg.TRAINLOG.DATA_NAMES[kk][0]].append(exel_out)

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





