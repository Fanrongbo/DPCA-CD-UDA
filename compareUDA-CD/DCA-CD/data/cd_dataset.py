import os.path
import torch
from data.image_folder import make_dataset
from data.preprocessing import Preprocessing
from PIL import Image
import numpy as np
from option.config import cfg


class ChangeDetectionDataset(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ### input T1_img
        if opt.phase in ['train','val']:
            dir_t1 = 'T1'
            self.dir_t1 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s], opt.phase, dir_t1)
            self.t1_paths = sorted(make_dataset([self.dir_t1]))

            ### input T2_img
            dir_t2 = 'T2'
            self.dir_t2 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s], opt.phase, dir_t2)
            self.t2_paths = sorted(make_dataset([self.dir_t2]))

            ### input change_label
            dir_label = 'label'
            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s], opt.phase, dir_label)
            self.label_paths = sorted(make_dataset([self.dir_label]))
        elif opt.phase in ['valTr']:
            dir_t1 = 'T1'
            self.dir_t1 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s], 'val', dir_t1)
            self.t1_paths = sorted(make_dataset([self.dir_t1]))

            ### input T2_img
            dir_t2 = 'T2'
            self.dir_t2 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s], 'val', dir_t2)
            self.t2_paths = sorted(make_dataset([self.dir_t2]))

            ### input change_label
            dir_label = 'label'
            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s], 'val', dir_label)
            self.label_paths = sorted(make_dataset([self.dir_label]))
        elif opt.phase in ['targetSelect']:
            dir_t1 = 'T1'
            self.dir_t1 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'train', dir_t1)
            # list_path = opt.hardspilt.format('all')
            ### input T2_img
            dir_t2 = 'T2'
            self.dir_t2 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'train', dir_t2)
            # self.t2_paths = sorted(make_dataset([self.dir_t2]))
            dir_label = 'label'
            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'train', dir_label)
            self.t1_paths=[]
            self.t2_paths=[]
            self.label_paths=[]
            # with open(list_path) as f:
            #     for i_id in f:
            #         i_id=i_id.strip()
            #         self.t1_paths.append(i_id)
            #         self.t2_paths.append(i_id.replace('T1', 'T2'))
            #         self.label_paths.append(i_id.replace('T1', 'label'))

            ### input change_label
            # self.t1_paths = sorted(make_dataset([self.dir_t1]))
            # self.t2_paths = sorted(make_dataset([self.dir_t2]))
            # self.label_paths = sorted(make_dataset([self.dir_label]))
        else:
            dir_t1 = 'T1'
            self.dir_t1_1 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'train', dir_t1)
            self.dir_t1_2 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'val', dir_t1)
            self.t1_paths = sorted(make_dataset([self.dir_t1_1,self.dir_t1_2]))

            ### input T2_img
            dir_t2 = 'T2'
            self.dir_t2_1 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'train', dir_t2)
            self.dir_t2_2 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'val', dir_t2)
            self.t2_paths = sorted(make_dataset([self.dir_t2_1,self.dir_t2_2]))

            ### input change_label
            dir_label = 'label'
            self.dir_label_1 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'train', dir_label)
            self.dir_label_2 = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'val', dir_label)
            self.label_paths = sorted(make_dataset([self.dir_label_1,self.dir_label_2]))


        self.dataset_size = len(self.t1_paths)
        self.preprocess = Preprocessing(
            img_size=self.opt.img_size,
            with_Lchannel=opt.LChannel
        )
        # if self.opt.phase == 'train':
        #     print('with_Lchannel=opt.LChannel',opt.LChannel)
        #     self.preprocess = Preprocessing(
        #                                     img_size=self.opt.img_size,
        #                                     with_random_hflip=opt.aug,
        #                                     with_random_vflip=opt.aug,
        #                                     with_scale_random_crop=opt.aug,
        #                                     with_random_blur=opt.aug,
        #                                     with_Lchannel=opt.LChannel
        #                                     )
        # else:
        #     self.preprocess= Preprocessing(
        #                                     img_size=self.opt.img_size,
        #                                     with_Lchannel=opt.LChannel
        #                                     )

    def __getitem__(self, index):
        ### input T1_img 
        t1_path = self.t1_paths[index]
        t1_img = np.asarray(Image.open(t1_path).convert('RGB'))
        # print('t1_path',t1_path,t1_img.shape)
        # print(t1_img.shape)
        if t1_img.shape[0]<self.opt.img_size or t1_img.shape[1]<self.opt.img_size:
            # print(t1_img.shape)
            t1_img = np.resize(t1_img, (self.opt.img_size, self.opt.img_size,3))
        # t1_img=np.resize(t1_img,(self.opt.img_size,self.opt.img_size,3))
        ### input T2_img
        t2_path = self.t2_paths[index]
        t2_img = np.asarray(Image.open(t2_path).convert('RGB'))
        if t2_img.shape[0]<self.opt.img_size or t2_img.shape[1]<self.opt.img_size:
            t2_img = np.resize(t2_img, (self.opt.img_size, self.opt.img_size,3))
        # t2_img=np.resize(t2_img,(self.opt.img_size,self.opt.img_size,3))

        ### input label
        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path), dtype=np.uint8)
        # print('label',label.shape)
        if len(label.shape)==3:
            label=label[:,:,0]
        if label.shape[0]<self.opt.img_size or label.shape[1]<self.opt.img_size:
            label = np.resize(label, (self.opt.img_size, self.opt.img_size))

        if self.opt.label_norm == True:
            label = label // 255
        # print(t1_path)
        ### transform
        [t1_tensor, t2_tensor], [label_tensor] = self.preprocess.transform([t1_img, t2_img], [label], to_tensor=True)

        input_dict = {'t1_img': t1_tensor, 't2_img': t2_tensor, 'label': label_tensor,
                      't1_path': t1_path, 't2_path': t2_path, 'label_path': label_path}

        return input_dict

    def __len__(self):
        return len(self.t1_paths) // self.opt.batch_size * self.opt.batch_size
