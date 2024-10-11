import torch.utils.data
from data.cd_dataset import ChangeDetectionDataset
from option.config import cfg

def CreateDataset(opt):
    dataset = None

    dataset = ChangeDetectionDataset()
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(torch.utils.data.Dataset):

    def initialize(self, opt):
        if opt.phase!='train' and opt.phase!='target_train' and opt.phase!='valTr'and opt.phase!='target':
            size=1
        else:
            size=opt.batch_size
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=size,
            shuffle=(opt.phase == 'train')or(opt.phase == 'target_train')or(opt.phase == 'valTr'),
            # shuffle=False,
            num_workers=int(opt.num_threads))


    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader