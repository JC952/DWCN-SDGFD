import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import scipy.io as scio
from scipy.fftpack import fft
import numpy as np

data_pth={
          "GearBox.BJUT":'.\data\GearBox.BJUT\GearData',
          "Bearing.BJTU":'.\data\Bearing.BJTU\BearingData',
          }

classes_map = {
    'GearBox.BJUT': 5,
    'Bearing.BJTU': 5,
}
domain_map = {
    'GearBox.BJUT': [1200,1800,2400,3000],
    'Bearing.BJTU': [1200,1210,2400,2410,3600,3610],

}
def get_domain_file(name):
    if name not in data_pth:
        raise ValueError('Name of datasetpu unknown %s' %name)
    return data_pth[name]
def get_domain_task(name):
    if name not in domain_map.keys():
        raise ValueError('Name of datasetpu unknown %s' %name)
    return domain_map[name]

class Dataset_data(Dataset):
    def __init__(self, data_tensor, target_tensor,train_y_domain_label):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.target_domain = train_y_domain_label

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index],self.target_domain[index]
class Fault_dataset(Dataset):
    def __init__(self,args):
        self.args = args
        self.data_pth =get_domain_file(self.args.dataset_name)
        self.task=get_domain_task(self.args.dataset_name)
        self.n_class=classes_map[self.args.dataset_name]-len(self.args.miss_class)

    def load_data(self,path, temp, sum_class,data_ratio, mis_class, FFT=True, normalize_type="0-1"):
        data_temp = scio.loadmat(path)
        data = data_temp.get(temp)
        self.n_class = int(sum_class) - len(mis_class)
        for i_c in mis_class:
            mis_class_id = np.argwhere(data[:, -1] == i_c)
            data = np.delete(data, mis_class_id, axis=0)
        class_sample, _ = data.shape
        train_x, test_x, train_y, test_y = train_test_split(data[:, :1024], data[:, -1], test_size=data_ratio)
        if FFT:
            train_x = np.abs(fft(train_x, axis=1))[:, :1024]
            test_x =np.abs(fft(test_x , axis=1))[:, :1024]
        train_x = self.Normalize(train_x, normalize_type)
        test_x = self.Normalize(test_x, normalize_type)
        train_x = torch.FloatTensor(train_x).unsqueeze(1)
        train_y = torch.LongTensor(train_y)
        test_x = torch.FloatTensor(test_x).unsqueeze(1)
        test_y = torch.LongTensor(test_y)
        return train_x, train_y, test_x, test_y

    def Normalize(self,data, type):
        seq = data
        if type == "0-1":
            Zmax, Zmin = seq.max(axis=1), seq.min(axis=1)
            seq = (seq - Zmin.reshape(-1, 1)) / (Zmax.reshape(-1, 1) - Zmin.reshape(-1, 1))
        elif type == "1-1":  # 把值变到[-1,1]
            seq = 2 * (seq - seq.min()) / (seq.max() - seq.min()) + -1
        elif type == "mean-std":
            mean = np.mean(seq, axis=1, keepdims=True)
            std = np.std(seq, axis=1, keepdims=True)
            std[std == 0] = 1
            seq = (seq - mean) / std
        else:
            seq=seq
        return seq


    def Loader(self,data_list_name=[],train=False,miss_class=[]):
        train_loader_x = []
        test_loader_x = []
        for domain_id,domain in enumerate(data_list_name):
            sum_class = classes_map[self.args.dataset_name]
            root=data_pth[self.args.dataset_name]+"_"+str(domain)+"_"+str(sum_class)+".mat"
            temp= root.split('\\')[-1].split('.')[0]
            train_x, train_y, test_x, test_y= self.load_data(root,temp, sum_class,self.args.data_ratio, miss_class, self.args.FFT, self.args.normalize_type)
            train_domain=torch.full_like(train_y,domain_id)
            train_dataset = Dataset_data(train_x, train_y,train_domain)
            test_domain=torch.full_like(test_y,domain_id)
            test_dataset = Dataset_data(test_x, test_y,test_domain)
            batch_size=self.args.batch_size
            if train:
                train_loader_x = Data.DataLoader(train_dataset, batch_size, drop_last=True)
                test_loader_x = Data.DataLoader(test_dataset, batch_size, drop_last=True)
            else:
                train_loader_x.append(train_dataset)
                test_loader_x.append(test_dataset)
        return train_loader_x, test_loader_x

    def task_loaders(self,dataset):
        """Prepares task mapping and loaders for each combination of source and target domains."""
        task_mapping = {}
        for task_id, source_name in enumerate(dataset.task):
            source_list_name = list(filter(lambda x: x != source_name, dataset.task))
            a = str(source_name)
            task_mapping[a] = source_list_name
        return task_mapping



