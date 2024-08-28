import torch
import numpy as np
import pandas as pd
import pickle
class MovieLens1MDataset(torch.utils.data.Dataset):
    """
    MovieLens 1M Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    :param dataset_path: MovieLens dataset path
    https://github.com/rixwew/pytorch-fm/blob/master/torchfm/dataset/movielens.py
    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__()
        ratings_info = pd.read_csv(dataset_path + '/ratings.dat', sep='::', engine='python', header=None)
        self.items = ratings_info.iloc[:, :2]  # -1 because ID begins from 1
        self.targets = ratings_info.iloc[:, 2].to_numpy()
        users_info = pd.read_csv(dataset_path + '/users.dat', sep='::', engine='python', header=None)
        self.items = self.items.merge(users_info.iloc[:, :4], on=0)
        self.items.iloc[:, :2] -= 1  # -1 because ID begins from 1
        self.items['1_y'][self.items['1_y'] == 'M'] = 0
        self.items['1_y'][self.items['1_y'] == 'F'] = 1
        self.items = self.items.to_numpy().astype(np.int)
        self.targets = self.__preprocess_target(self.targets).astype(np.float32)

        # 5_field_dims: user_id,item_id,gender,age,occupation
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return
    

class PrecessMLtag():
    def __init__(self):
        dataset_path = "."
        self.ratings_info = pd.read_csv(dataset_path + '/ratings.dat', sep='::', engine='python', header=None)
        self.items = self.ratings_info.iloc[:,:2]
        self.targets = self.ratings_info.iloc[:, 2]
        self.targets[self.targets <= 3] = 0
        self.targets[self.targets > 3] = 1
        # self.targets = self.targets.to_numpy().astype(np.float32)

        self.users_info = pd.read_csv(dataset_path + '/users.dat', sep='::', engine='python', header=None)
        self.items = self.items.merge(self.users_info.iloc[:, :4], on=0)
        # self.items.iloc[:, :2] -= 1  # -1 because ID begins from 1
        print(self.items.columns)
        self.items.columns = ["user_id","item_id","gender","age","occupation"]

        # self.items.columns = ["user_id","item_id","gender","age","occupation","gender1","age1","occupation1"]
        self.items['gender'][self.items['gender'] == 'M'] = 0
        self.items['gender'][self.items['gender'] == 'F'] = 1
        print(self.items.columns)
        self.construct_df()

    def construct_df(self):
        self.field_dims = []
        for i in self.items.columns:
            maps = {val: k for k, val in enumerate(set(self.items[i]))}
            self.items[i] = self.items[i].map(maps)
            # self.features_maps[i] = maps
            num = self.items[i].nunique()
            print(i,num)
            self.field_dims.append(num)
        # self.df_data[0] = self.df_data[0].apply(lambda x: max(x, 0))
        print(self.field_dims)
        pickle.dump((self.items,self.targets,self.field_dims), open("ctrml1m.p","wb"))



class RecML1M():
    def __init__(self, data, data_y):
        self.data = data
        self.data_y = data_y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values, self.data_y.iloc[idx]


def getdataloader_ml1m(path="", batch_size=1024, prefix="./"):
    path_ml = path + "ctrml1m.p" # The dataset has been preprocessed.  
    data_x, data_y, field_dims = pickle.load(open(path_ml, mode="rb"))
    all_length = len(data_x)
    print("all_length", all_length)
    # [6040, 3706, 2, 7, 21]
    print(field_dims)
    print(sum(field_dims))
    # 8:1:1
    valid_size = int(0.1 * all_length)
    train_size = all_length - valid_size

    print("all", train_size + valid_size)
    train_dataset = RecML1M(data_x, data_y)
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size - valid_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("train_loader", train_loader.__len__())
    print("valid_loader", valid_loader.__len__())
    print("test_loader", test_loader.__len__())

    return field_dims, train_loader, valid_loader, test_loader


if __name__ == '__main__':
    import os 
    print(os.getcwd())
    fields, train_loader, valid_loader, test_loader = getdataloader_ml1m(batch_size=128)
    one_iter = iter(test_loader)
    nexts = one_iter.__next__()
    print(nexts)
    print(fields)
    print(sum(fields))