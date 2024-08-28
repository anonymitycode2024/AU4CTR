import pandas as pd
import torch
import os


class LoadData811():
    def __init__(self, path="./", dataset="ml", loss_type="square_loss"):
        self.dataset = dataset
        self.loss_type = loss_type
        self.path = path
        self.trainfile = self.path  + "ml.train.libfm"
        self.testfile = self.path  + "ml.test.libfm"
        self.validationfile = self.path + "ml.validation.libfm"
        self.features_M = {}
        self.construct_df()
    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")
        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        # self.train_valid = pd.concat([self.data_train, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            # if self.dataset != "frappe":
            # maps = {}
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.data_test[i] = self.data_test[i].map(maps)
            self.data_train[i] = self.data_train[i].map(maps)
            self.data_valid[i] = self.data_valid[i].map(maps)
            
            self.all_data[i] = self.all_data[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))
        # self.all_data[0] = self.all_data[0].apply(lambda x: max(x, 0))
        self.data_test[0] = self.data_test[0].apply(lambda x: max(x, 0))
        self.data_train[0] = self.data_train[0].apply(lambda x: max(x, 0))
        self.data_valid[0] = self.data_valid[0].apply(lambda x: max(x, 0))

class MlTagData(torch.utils.data.Dataset):
    def __init__(self, all_data):
        self.all_data = all_data


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        x = self.all_data.iloc[idx].values[1:]
        y1 = self.all_data.iloc[idx].values[0]
        return x, y1


def get_mltag_loader721(path="./", dataset="movielens", num_ng=4, batch_size=256):
    AllDataF = LoadData811(path=path, dataset=dataset)
    data_train = MlTagData(AllDataF.data_train)
    data_valid = MlTagData(AllDataF.data_valid)
    data_test = MlTagData(AllDataF.data_test)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))
    return AllDataF.field_dims, train_loader, valid_loader, test_loader

if __name__ == '__main__':
    print(os.getcwd())
    fields, train_loader, valid_loader, test_loader = get_mltag_loader721(batch_size=128)
    one_iter = iter(test_loader)
    nexts = one_iter.__next__()
    print(nexts)
    print(fields)
    print(sum(fields))