import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchinfo import summary
import datetime
import yaml
import json
import argparse
import logging
from utils import StandardScaler, DataLoader, masked_mae_loss, masked_mape_loss, \
    masked_rmse_loss, print_log, CustomJSONEncoder, quadruplet_loss, steps_output
from MTGRN import MTGRN

def prepare_x_y(x, y, cfg):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    :return : x , y ,y_cov 时间信息
    """
    input_dim = cfg['model_args']['input_dim']
    output_dim = cfg['model_args']['output_dim']
    x_input_dim = x[..., :input_dim]
    x_time = x[..., input_dim:]
    y_input_dim = y[..., :output_dim]
    y_time = y[..., output_dim:]
    x_input_dim = torch.from_numpy(x_input_dim).float()
    x_time = torch.from_numpy(x_time).float()
    y_input_dim = torch.from_numpy(y_input_dim).float()
    y_time = torch.from_numpy(y_time).float()
    return x_input_dim.to(DEVICE), y_input_dim.to(DEVICE), x_time.to(DEVICE), y_time.to(DEVICE)

def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_log(name, param.shape, param.numel(), log=log)
            param_count += param.numel()
    print_log(f'In total: {param_count} trainable parameters.', log=log)
    return

def load_dataset(args, cfg, log=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(f'../{args.dataset}', category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], cfg['batch_size'], shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], cfg['batch_size'], shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], cfg['batch_size'], shuffle=False)

    print_log('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape, log=log)
    print_log('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape, log=log)
    print_log('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape, log=log)

    return data, scaler
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def evaluate(model, data, mode=None):
    with torch.no_grad():
        model = model.eval()
        data_iter = data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        maes, mapes, rmses = [], [], []
        l_3, m_3, r_3 = [], [], []
        l_6, m_6, r_6 = [], [], []
        l_12, m_12, r_12 = [], [], []
        for x, y in data_iter:
            x, y, x_cov, ycov = prepare_x_y(x, y, cfg)
            output, query_s, pos_s, neg_s1, neg_s2, query_t, pos_t, neg_t1, neg_t2 = model(x, x_cov, ycov)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss_all = masked_mae_loss(y_pred, y_true)  # masked_mae_loss(y_pred, y_true)

            loss_s = quadruplet_loss(query_s, pos_s.detach(), neg_s1.detach(), neg_s2.detach())
            loss_t = quadruplet_loss(query_t, pos_t.detach(), neg_t1.detach(), neg_t2.detach())

            compact_loss = nn.MSELoss()
            loss_sc = compact_loss(query_s, pos_s.detach())
            loss_tc = compact_loss(query_t, pos_t.detach())

            loss = loss_all + cfg['lamb1'] * loss_t + cfg['lamb1'] * loss_tc + cfg['lamb2'] * loss_s + cfg['lamb2'] * loss_sc
            losses.append(loss.item())
            # Followed the DCRNN TensorFlow Implementation
            maes.append(masked_mae_loss(y_pred, y_true).item())
            mapes.append(masked_mape_loss(y_pred, y_true).item())
            rmses.append(masked_rmse_loss(y_pred, y_true).item())

            y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
            l_3, m_3, r_3, l_6, m_6, r_6, l_12, m_12, r_12 = steps_output(y_true, y_pred, l_3, m_3, r_3, l_6, m_6, r_6, l_12, m_12, r_12)
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        mean_mae, mean_mape, mean_rmse = np.mean(maes), np.mean(mapes), np.sqrt(np.mean(rmses))
        l_3, m_3, r_3 = np.mean(l_3), np.mean(m_3), np.sqrt(np.mean(r_3))
        l_6, m_6, r_6 = np.mean(l_6), np.mean(m_6), np.sqrt(np.mean(r_6))
        l_12, m_12, r_12 = np.mean(l_12), np.mean(m_12), np.sqrt(np.mean(r_12))
        if mode == 'test':
            print_log(
                'Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_mae, mean_mape, mean_rmse), log=log)
            print_log('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_3, m_3, r_3), log=log)
            print_log('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_6, m_6, r_6), log=log)
            print_log('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_12, m_12, r_12), log=log)
            print_log(log=log)
        return mean_loss, ys_true, ys_pred

def train_one_epoch(model, batches_seen, optimizer, lr_scheduler, data, scaler, cfg, log=None):
    model = model.train()
    data_iter = data['train_loader'].get_iterator()
    losses = []

    for x, y in data_iter:
        optimizer.zero_grad()
        x, y, x_cov, ycov = prepare_x_y(x, y, cfg)
        # x.shape: 64, 12, node, 1 # speed/flow
        # x_cov.shape: 64, 12, node, 2 # time
        # y.shape: 64, 12, node, 1 # speed/flow
        # ycov.shape: 64, 12, node, 2 # time
        output, query_s, pos_s, neg_s1, neg_s2, query_t, pos_t, neg_t1, neg_t2 = model(x, x_cov, ycov, y, batches_seen)
        y_pred = scaler.inverse_transform(output)
        y_true = scaler.inverse_transform(y)

        loss_all = masked_mae_loss(y_pred, y_true)  # masked_mae_loss(y_pred, y_true)

        loss_s = quadruplet_loss(query_s, pos_s.detach(), neg_s1.detach(), neg_s2.detach())
        loss_t = quadruplet_loss(query_t, pos_t.detach(), neg_t1.detach(), neg_t2.detach())

        compact_loss = nn.MSELoss()
        loss_sc = compact_loss(query_s, pos_s.detach())
        loss_tc = compact_loss(query_t, pos_t.detach())

        loss = loss_all + cfg['lamb1'] * loss_t + cfg['lamb1'] * loss_tc + cfg['lamb2'] * loss_s + cfg['lamb2'] * loss_sc
        losses.append(loss.item())
        batches_seen += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       cfg['max_grad_norm'])  # gradient clipping - this does it in place
        optimizer.step()
    train_loss = np.mean(losses)
    return train_loss, batches_seen

def traintest_model(model, optimizer, lr_scheduler, data, scaler, cfg, model_path, log=None):
    model = model.to(DEVICE)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(cfg['epochs']):
        start_time = time.time()
        train_loss, batches_seen = train_one_epoch(model, batches_seen, optimizer, lr_scheduler, data, scaler, cfg, log=None)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, data, mode='val')
        # if (epoch_num % args.test_every_n_epochs) == args.test_every_n_epochs - 1:
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1,
                                                                                                        cfg['epochs'],
                                                                                                        batches_seen,
                                                                                                        train_loss,
                                                                                                        val_loss,
                                                                                                        optimizer.param_groups[
                                                                                                            0]['lr'], (
                                                                                                                end_time2 - start_time))
        print_log(message, log=log)
        print_log(log=log)
        test_loss, _, _ = evaluate(model, data, 'test')
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            # logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == cfg['early_stop']:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break

    print_log('=' * 35 + 'Best model performance' + '=' * 35, log=log)
    print_log(log=log)
    model = model
    model.load_state_dict(torch.load(model_path))
    test_loss, _, _ = evaluate(model, data, 'test')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS08')
    parser.add_argument('--g', type=int, default=0)
    args = parser.parse_args()
    

    GPU_ID = args.g
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()

    model_name = MTGRN.__name__
    with open(f'{model_name}.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]
    # -------------------------- seed ------------------------- #
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed(cfg['seed'])
    # -------------------------- load model ------------------------- #
    model = MTGRN(**cfg['model_args'])
    # ----------------------------make log file----------------------------- #
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_path = f'../logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f'{model_name}-{args.dataset}-{now}.log')
    log = open(log, 'a')
    log.seek(0)
    log.truncate()

    # ----------------------------load dataset ---------------------------- #
    print_log(dataset, log=log)
    data, scaler = load_dataset(args, cfg, log=log)
    print_log(log=log)

    # ----------------------------save model ---------------------------- #
    path = f'../saved_models/{dataset}_{model_name}_{now}'   # filename
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, f'{model_name}-{now}.pt')
    shutil.copy2(sys.argv[0], path)
    shutil.copy2(f'{model_name}.py', path)
    shutil.copy2('utils.py', path)

    # ----------------------------set model opt, scheduler ---------------------------- #
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=cfg['epsilon'], weight_decay=cfg['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['lr_decay_ratio'])

    # ----------------------------print model structure---------------------------- #
    print_log("-----------", model_name, "-----------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log("----------------------------------------\n", log=log)
    print_model(model)
    # ----------------------------train and test model ---------------------------- #
    traintest_model(model, optimizer, lr_scheduler, data, scaler, cfg, model_path, log=log)



