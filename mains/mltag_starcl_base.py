import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import random
import sys
import tqdm
import time
import argparse
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import log_loss, roc_auc_score

sys.path.append("..")
from model.models import *
from model.AU4CTR import StarCL
from data import get_mltag_loader721
from utils.utils_de import *
from utils.earlystoping import EarlyStopping, EarlyStoppingLoss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(
        name,
        field_dims,
        batch_size=4096,
        pratio=0.5,
        embed_dim=20,
        mlp_layers=(400, 400, 400)):
    if name == "fm_s":
        return FM_CL(field_dims=field_dims, embed_dim=embed_dim, pratio=pratio)
    
    elif name == "fm_cl4":
        return FM_CL4CTR(field_dims, embed_dim, batch_size=batch_size, pratio=pratio)

    elif name == "dcnv2_s":
        return DCNV2P_CL(field_dims, embed_dim, pratio=pratio)

    elif name == "dcn_s":
        return DCN_CL(field_dims, embed_dim, pratio=pratio)

    elif name == "dfm_s":
        return DeepFM_CL(field_dims, embed_dim,pratio=pratio)

    else:
        raise ValueError('unknown model name: ' + name)


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


def train(model,
          optimizer,
          data_loader,
          criterion,
          lambda_au=1.0,
          beta=1e-2,
          lambda_i=1e-2):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader)):
        label = label.float()
        user_item = user_item.long()
        user_item = user_item.cuda()
        label = label.cuda()

        model.zero_grad()

        pred_y = torch.sigmoid(model(user_item).squeeze(1))
        loss_y = criterion(pred_y, label)
        loss = loss_y + model.compute_cl_loss_all(user_item, lambda_au=lambda_au, beta=beta, lambda_i=lambda_i)
        # loss = loss_y 
        # loss = loss_y + model.compute_cl_loss_two(lambda_au=lambda_au, beta=beta)
        # loss = loss_y + model.compute_cl_loss_self(user_item,  lambda_i=lambda_i)
        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()
        
    ave_loss = total_loss / (i + 1)
    return ave_loss


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            fields = fields.long()
            target = target.float()
            fields, target = fields.cuda(), target.cuda()
            y = torch.sigmoid(model(fields).squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name, model_name, epoch, embed_dim, learning_rate,
         batch_size, weight_decay, save_dir, path,
         pratio, lambda_au, beta, lambda_i, hint):
    field_dims, trainLoader, validLoader, testLoader = \
        get_mltag_loader721(path="../data/mltag/", batch_size=batch_size)
    print(field_dims)
    time_fix = time.strftime("%m%d%H%M%S", time.localtime())

    for K in [embed_dim]:
        paths = os.path.join(save_dir, dataset_name, model_name, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths)
        with open(paths + f"/{model_name}_{K}_{batch_size}_{lambda_au}_{beta}_{lambda_i}_{pratio}_{time_fix}.p",
                  "a+") as fout:
            fout.write("Batch_size:{}\tembed_dim:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\tpratio:{}\t"
                       "\tlambda_au:{}\tbeta:{}\lambda_i:{}\n"
                       .format(batch_size, K, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay,
                               pratio, lambda_au, beta,lambda_i))
            print("Start train -- K : {}".format(K))

            criterion = torch.nn.BCELoss()
            model = get_model(
                name=model_name,
                field_dims=field_dims,
                batch_size=batch_size,
                embed_dim=K,
                pratio=pratio).cuda()

            params = count_params(model)
            fout.write("hint:{}\n".format(hint))
            fout.write("count_params:{}\n".format(params))
            print(params)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

            early_stopping = EarlyStoppingLoss(patience=6, verbose=True, prefix=path)
            scheduler_min = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)
            val_auc_best = 0
            auc_index_record = ""

            val_loss_best = 1000
            loss_index_record = ""

            for epoch_i in range(epoch):
                print(__file__, model_name, K, epoch_i, "/", epoch)
                print("Batch_size:{}\tembed_dim:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\tpratio:{}\t"
                      "\tlambda_au:{}\tbeta:{}\tlambda_i:{}\t"
                      .format(batch_size, K, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay,
                              pratio, lambda_au, beta, lambda_i))
                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion, lambda_au=lambda_au, beta=beta, lambda_i=lambda_i)
                val_auc, val_loss = test_roc(model, validLoader)
                test_auc, test_loss = test_roc(model, testLoader)

                scheduler_min.step(val_loss)
                end = time.time()
                if val_loss < val_loss_best:
                    torch.save(model, paths + f"/{model_name}_best_auc_{K}_{pratio}_{time_fix}.pkl")

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                # early_stopping(val_auc)
                early_stopping(val_loss)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mltag', help="")
    parser.add_argument('--save_dir', default='../chkpt/chkpt_mltag/au4ctr', help="") 
    
    parser.add_argument('--path', default="../data/", help="")
    parser.add_argument('--model_name', default='fm', help="")
    parser.add_argument('--epoch', type=int, default=50, help="")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4096, help="batch_size")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="")
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--choice', default=0, type=int, help="choice")
    parser.add_argument('--hint', default="starcl", help="")
    parser.add_argument('--embed_dim', default=16, type=int, help="the size of feature dimension")

    parser.add_argument('--pratio', default=0.3, type=float, help="pratio")
    parser.add_argument('--lambda_au', default=1e-0, type=float, help="lambda_au for alignment loss")
    parser.add_argument('--beta', default=1e-2, type=float, help="beta for uniformity loss")
    parser.add_argument('--lambda_i', default=1e-2, type=float, help="lambda_i for ctr interaction alignment loss")
    args = parser.parse_args()

    if args.choice == 0:
        model_names = ["dcnv2_s"] * 3
    
    print(model_names)
    for batch_size in [4096]:
        for learning_rate in [1e-3]:
            for weight_decay in [1e-4]:
                for embed_dim in [16]: 
                    for lambda_au in [0.0]:
                        for beta in [0.0]:
                            for lambda_i in [0.0]:
                                for pratio in [0.0]:
                                    for name in model_names:
                                        main(dataset_name=args.dataset_name,
                                            model_name=name,
                                            epoch=args.epoch,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size,
                                            weight_decay=weight_decay,
                                            save_dir=args.save_dir,
                                            path=args.path,
                                            pratio=pratio,
                                            embed_dim=embed_dim,
                                            lambda_au=lambda_au,
                                            beta=lambda_au,
                                            lambda_i=lambda_i,
                                            hint=args.hint
                                            )