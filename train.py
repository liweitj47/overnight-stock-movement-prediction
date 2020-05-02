from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import utils
import lr_scheduler as L
from models import *
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
import random
from Data import *
import numpy as np
import math

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)


# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore', type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-use_medium', default=False, action='store_true',
                        help="whether to use the third class medium during training")
    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-multi_turn', default=False, action='store_true',
                        help="train multiple times and test")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')

    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = utils.read_config('config.yaml')
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def train(model, dataloader, scheduler, optim, updates):
    scores = []
    score = 0.
    max_acc = 0.
    adjs = dataloader.adjs
    print('training')
    for epoch in range(1, config.epoch + 1):
        total_right = 0
        total_num = 0
        total_loss = 0.
        total_tpx_right = 0
        total_tpx_num = 0
        start_time = time.time()

        if config.schedule:
            scheduler.step()
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        elif config.learning_rate_decay < 1:
            optim.updateLearningRate(score, epoch)
        model.train()

        train_data = dataloader.train
        random.shuffle(train_data)
        for span in tqdm(train_data, disable=not args.verbose):
            model.zero_grad()
            span_nodes, bert_vec, node_text, word_mask, sent_mask, node_features, last_movement, tpx_movement, movement_mask, news_mask, movement_num = span
            if movement_num == 0:
                continue
            if use_cuda:
                if config.hierarchical:
                    node_text = [t.cuda() for t in node_text]
                    word_mask = [m.cuda() for m in word_mask]
                    sent_mask = sent_mask.cuda()
                else:
                    node_text = node_text.cuda()
                    word_mask = word_mask.cuda()
                node_features = node_features.cuda()
                bert_vec = bert_vec.cuda()
                adjs = [adj.cuda() for adj in adjs]
                span_nodes = span_nodes.cuda()
                last_movement = last_movement.cuda()
                movement_mask = movement_mask.cuda()
                news_mask = news_mask.cuda()
                tpx_movement = tpx_movement.cuda()
            # print('last movement', last_movement)
            last_output, g_output = model(span_nodes, bert_vec, node_text, word_mask, sent_mask, node_features, adjs)
            right_num = torch.sum(
                (last_output.max(-1)[1] == last_movement).long() * (movement_mask * news_mask).long()).item()
            tpx_right_num = torch.sum((g_output.max(-1)[1] == tpx_movement).long() * (tpx_movement != 2).long() * (
                    news_mask.sum(-1) > 0).long()).item()
            total_tpx_num += torch.sum((tpx_movement != 2).long() * (news_mask.sum(-1) > 0).long()).item()
            total_tpx_right += tpx_right_num
            total_num += (movement_mask * news_mask).long().sum().item()
            total_right += right_num
            if args.use_medium:
                weight = torch.Tensor([1, 1, 1])
            else:
                weight = torch.Tensor([1, 1, 0])
            if use_cuda:
                weight = weight.cuda()
            last_loss = F.cross_entropy(last_output, last_movement, weight=weight,
                                        reduction='none') * movement_mask * news_mask
            loss = last_loss.sum()
            loss.backward()
            total_loss += loss.data.item()

            optim.step()
            updates += 1  # 进行了一次更新

        print('train total movement number', total_num)
        # logging中记录的是每次更新时的epoch，time，updates，correct等基本信息.
        # 还有score分数的信息
        tpx_acc = total_tpx_right / float(total_tpx_num)
        logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f, tpx acc: %.3f\n"
                % (time.time() - start_time, epoch, updates, total_loss,
                   total_right / float(total_num), tpx_acc*100))
        score = eval(model, dataloader, epoch, updates, do_test=False)
        scores.append(score)
        if score >= max_acc:
            save_model(log_path + 'best_model_checkpoint.pt', model, optim, updates)
            max_acc = score

        model.train()

    model = load_model(log_path + 'best_model_checkpoint.pt', model)
    test_acc = eval(model, dataloader, -1, -1, True)
    return max_acc, test_acc


def eval(model, dataloader, epoch, updates, do_test=False):
    model.eval()
    total_right = 0
    total_tpx_right = 0
    total_propagated_right = 0
    total_num = 0
    total_tpx_num = 0
    total_propagated_num = 0
    total_tp = 0
    total_fp = 0
    if do_test:
        data = dataloader.test
    else:
        data = dataloader.dev
    adjs = dataloader.adjs
    for span in tqdm(data, disable=not args.verbose):
        model.zero_grad()
        span_nodes, bert_vec, node_text, word_mask, sent_mask, node_features, last_movement, tpx_movement, movement_mask, news_mask, movement_num = span
        if movement_num == 0:
            continue
        if use_cuda:
            if config.hierarchical:
                node_text = [t.cuda() for t in node_text]
                word_mask = [m.cuda() for m in word_mask]
                sent_mask = sent_mask.cuda()
            else:
                node_text = node_text.cuda()
                word_mask = word_mask.cuda()
            bert_vec = bert_vec.cuda()
            node_features = node_features.cuda()
            adjs = [adj.cuda() for adj in adjs]
            span_nodes = span_nodes.cuda()
            last_movement = last_movement.cuda()
            movement_mask = movement_mask.cuda()
            news_mask = news_mask.cuda()
            tpx_movement = tpx_movement.cuda()
        # print('last movement', last_movement)
        last_output, g_output = model(span_nodes, bert_vec, node_text, word_mask, sent_mask, node_features, adjs)
        right_num = torch.sum(
            (last_output.max(-1)[1] == last_movement).long() * (movement_mask * news_mask).long()).item()
        tpx_right_num = torch.sum((g_output.max(-1)[1] == tpx_movement).long() * (tpx_movement != 2).long() * (
                    news_mask.sum(-1) > 0).long()).item()
        if len(adjs) > 1:
            propagated_mask = (torch.sparse.mm(adjs[0] + adjs[1], news_mask.unsqueeze(-1)) * (
                    1 - news_mask) > config.edge_threshold).float()
        else:
            propagated_mask = (torch.sparse.mm(adjs[0], news_mask.unsqueeze(-1)) * (
                    1 - news_mask) > config.edge_threshold).float()
        propagated_right_num = torch.sum(
            (last_output.max(-1)[1] == last_movement).float() * movement_mask * propagated_mask).item()
        tp = torch.sum((last_output.max(-1)[1] == last_movement).long() * (
                movement_mask * news_mask).long() * last_movement).item()
        fp = torch.sum((last_output.max(-1)[1] != last_movement).long() * (
                movement_mask * news_mask).long() * last_movement).item()
        # print('last movement', last_movement)
        total_num += (movement_mask * news_mask).long().sum().item()
        total_tpx_num += torch.sum((tpx_movement != 2).long() * (news_mask.sum(-1) > 0).long()).item()
        total_propagated_num += (movement_mask * propagated_mask).long().sum().item()
        total_tp += tp
        total_fp += fp
        total_right += right_num
        total_tpx_right += tpx_right_num
        total_propagated_right += propagated_right_num
    total_false = total_num - total_right
    total_tn = total_right - total_tp
    total_fn = total_false - total_fp
    news_acc = total_right / float(total_num)
    tpx_acc = total_tpx_right / float(total_tpx_num)
    acc_propagated = total_propagated_right / float(total_propagated_num)
    total_acc = (total_right + total_propagated_right) / (total_num + total_propagated_num)
    if (total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn) == 0:
        mcc = 0
    else:
        mcc = (total_tp * total_tn - total_fp * total_fn) / math.sqrt(
            float((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn)))
    logging_csv([epoch, updates, news_acc, acc_propagated, total_acc])
    logging('eval total movement number %d, news movement number %d, propagated number %d\n' % (
        total_num + total_propagated_num, total_num, total_propagated_num))
    logging('evaluating news accuracy %.2f, mcc %.4f\n' % (news_acc * 100, mcc))
    logging('evaluating tpx accuracy %.2f\n' % (tpx_acc * 100))
    logging('evaluating propagated accuracy %.2f, total acc%.2f\n' % (acc_propagated * 100, total_acc * 100))
    return news_acc


def save_model(path, model, optim, updates):
    '''保存的模型是一个字典的形式, 有model, config, optim, updates.'''

    # 如果使用并行的话使用的是model.module.state_dict()
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def load_model(path, model):
    checkpoints = torch.load(path)
    model.load_state_dict(checkpoints['model'])
    return model


def main(vocab, dataloader):
    # 设定种子
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # checkpoint
    if args.restore:  # 存储已有模型的路径
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))

    torch.backends.cudnn.benchmark = True

    # model
    print('building model...\n')
    # configure the model
    # Model and optimizer
    model = GLSTM(config, vocab)
    # model = hierarchical_attention(config, vocab)
    # model = SLSTM(config, vocab)
    # model = Transformer(config, vocab)
    if args.restore:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()
    if len(args.gpus) > 1:  # 并行
        model = nn.DataParallel(model, device_ids=args.gpus, dim=1)
    logging(repr(model) + "\n\n")  # 记录这个文件的框架

    # total number of parameters
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]

    logging('total number of parameters: %d\n\n' % param_count)

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
        ori_updates = updates
    else:
        updates = 0

    # optimizer
    if args.restore:
        optim = checkpoints['optim']
    else:
        optim = Optim(config.optim, config.learning_rate, config.max_grad_norm, lr_decay=config.learning_rate_decay,
                      start_decay_at=config.start_decay_at)

    optim.set_parameters(model.parameters())
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    else:
        scheduler = None

    if not args.notrain:
        max_acc, test_acc = train(model, dataloader, scheduler, optim, updates)
        logging("Best accuracy: %.2f, test accuracy: %.2f\n" % (max_acc * 100, test_acc * 100))
        return test_acc
    else:
        assert args.restore is not None
        eval(model, vocab, dataloader, 0, updates, do_test=True)


if __name__ == '__main__':
    # Load data
    start_time = time.time()
    vocab = Vocab(config.vocab_file, config.emb_file, config.emb_size, config.use_pre_emb, config.vocab_size)
    print('loading data...\n')
    dataloader = DataLoader(config, vocab, debug=args.debug)
    print("DATA loaded!")
    # data
    print('loading time cost: %.3f' % (time.time() - start_time))
    main(vocab, dataloader)
