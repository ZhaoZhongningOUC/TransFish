import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

'''
构建dataloader类加载数据
'''


class TrajDataset(Dataset):
    """
    下面这三个是必备函数，必须要写
    __init__负责读取
    __getitem__负责获取数据编号
    __len__返回总长度
    """
    def __init__(self, name, device):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(current_dir, '../../dataset/'))
        self.heatmap = torch.from_numpy(np.load(f'{file_path}/{name}/heatmap.npy').astype(np.float32))
        self.earlybird_heatmap = torch.from_numpy(np.load(f'{file_path}/{name}/earlybird.npy').astype(np.float32))
        self.ssh = torch.from_numpy(np.load(f'{file_path}/{name}/ssh.npy').astype(np.float32))
        self.sst = torch.from_numpy(np.load(f'{file_path}/{name}/sst.npy').astype(np.float32))
        self.sss = torch.from_numpy(np.load(f'{file_path}/{name}/sss.npy').astype(np.float32))
        self.curr = torch.from_numpy(np.load(f'{file_path}/{name}/cur.npy').astype(np.float32))
        self.cha = torch.from_numpy(np.load(f'{file_path}/{name}/cha.npy').astype(np.float32))
        self.device = device

    def __getitem__(self, index):

        return self.heatmap[index].to(self.device), \
               self.earlybird_heatmap[index].to(self.device), \
               self.ssh[index].to(self.device), \
               self.sst[index].to(self.device), \
               self.sss[index].to(self.device), \
               self.curr[index].to(self.device), \
               self.cha[index].to(self.device)

    def __len__(self):
        return self.heatmap.shape[0]


def get_dataloader(name, device, batch_size=32, shuffle=True, drop_last=True):
    """
    必备函数，改对应参数即可
    """
    dataset = TrajDataset(name, device)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last) # num_workers=8, pin_memory=True
    return data_loader


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if cuda else "cpu")
    dataLoader = get_dataloader('train', device, batch_size=32, shuffle=False, drop_last=True)
    print(len(dataLoader))
    dataLoader = get_dataloader('valid', device, batch_size=32, shuffle=False, drop_last=True)
    print(len(dataLoader))
    for heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha in dataLoader:
        print('heatmap', heatmap.shape)
        print('earlybird_heatmap', earlybird_heatmap.shape)
        print('ssh', ssh.shape)
        print('sst', sst.shape)
        print('sss', sss.shape)
        print('curr', curr.shape)
        print('cha', cha.shape)
        exit()
