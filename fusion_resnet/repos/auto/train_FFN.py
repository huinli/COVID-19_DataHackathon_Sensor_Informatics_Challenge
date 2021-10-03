from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models import resnet
from models import ffn as fn
from config import cfg, update_config
from utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset_mul import DeepSpeakerDataset
from functions import train_from_scratch, validate_identificate,train_ffn
from loss import CrossEntropyLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train energy network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--load_path',
                        help="The path to resumed dir",
			required=True,
                        type=str)

    parser.add_argument('--fold_num',
                        help="The folder number",
			required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)


    args = parser.parse_args()

    return args

def getmodels(args,cfg,path):

    model = eval('resnet.{}(num_classes={})'.format(
        cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()
    #checkpoint
    checkpoint = os.path.join(args.load_path, 'checkpoint','best_models',path)
    assert os.path.exists(checkpoint)
    _checkpoint = torch.load(checkpoint)
    model.load_state_dict(_checkpoint['state_dict'])
    
    #modify last layer
    
    #model = nn.Sequential(*list(model.children())[:-1])
    # optimizer
    optimizer= optim.Adam(
        model.net_parameters() if hasattr(model, 'net_parameters') else model.parameters(),
        lr=cfg.TRAIN.LR
    )
    return model, optimizer

def getData(cfg,path1,path2,path3):
    train_dataset = DeepSpeakerDataset(
        Path(path1),Path(path2),Path(path3),'dev', cfg.DATASET.PARTIAL_N_FRAMES)

    test_dataset_identification = DeepSpeakerDataset(
        Path(path1),Path(path2),Path(path3),'test', cfg.DATASET.PARTIAL_N_FRAMES, None, is_test=False)
    print(cfg.TRAIN.BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    test_loader_identification = torch.utils.data.DataLoader(
        dataset=test_dataset_identification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, test_loader_identification

def main():
    args = parse_args()
    update_config(cfg, args)
    proc_num = args.fold_num

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    
    # model and optimizer
    model_breath,optimizer_breath = getmodels(args,cfg,'breathing_{}.pth'.format(proc_num))
    model_cough,optimizer_cough = getmodels(args,cfg,'cough_{}.pth'.format(proc_num))
    model_speech, optimizer_speech = getmodels(args,cfg,'speech_{}.pth'.format(proc_num))
    
    ffn = fn.ffn()
    ffn = ffn.cuda()
    optimizer= optim.Adam(
        ffn.net_parameters() if hasattr(ffn, 'net_parameters') else ffn.parameters(),
        lr=cfg.TRAIN.LR
    )

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    best_acc1 = 0.0
    best_roc = 0.0
    last_epoch = -1

    # dataloader
    b_path = cfg.DATASET.DATA_DIR +'/breathing_{}_data'.format(proc_num)
    c_path = cfg.DATASET.DATA_DIR + '/cough_{}_data'.format(proc_num)
    s_path = cfg.DATASET.DATA_DIR + '/speech_{}_data'.format(proc_num)
    
    train_loader,test_loader_identification = getData(cfg,b_path,c_path,s_path)

    print("basic Dataset DIR: /"+cfg.DATASET.DATA_DIR)
    exp_name = args.cfg.split('/')[-1].split('.')[0]
    args.path_helper = set_path('logs', exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
    logger.info(cfg)
    logger.info("Number of parameters: {}".format(count_parameters(ffn)))
    # training setting
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // cfg.VAL_FREQ,
    }

    # training loop

    # Loss
    criterion = CrossEntropyLoss(cfg.MODEL.NUM_CLASSES).cuda()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR_MIN,
        last_epoch=last_epoch
    )

    for epoch in tqdm(range(begin_epoch, cfg.TRAIN.END_EPOCH), desc='train progress'):
        with torch.no_grad():
            model_breath.eval()
            model_cough.eval()
            model_speech.eval()
        
        train_ffn(cfg, model_breath,model_cough,model_speech,ffn, optimizer, train_loader, criterion, epoch, writer_dict, lr_scheduler)
        
        
        if epoch % cfg.VAL_FREQ == 0:
            acc, roc = validate_identificate(cfg, model_breath,model_cough,model_speech,ffn, test_loader_identification, criterion)

            # remember best acc@1 and save checkpoint
            is_best = roc > best_roc
            best_roc = max(roc, best_roc)
            best_acc1 = max(acc,best_acc1)

            # save
            logger.info('=> saving checkpoint to {}'.format(args.path_helper['ckpt_path']))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': ffn.state_dict(),
                'best_acc1': best_acc1,
                'best_roc':best_roc,
                'optimizer': optimizer.state_dict(),
                'path_helper': args.path_helper
            }, is_best, args.path_helper['ckpt_path'], 'checkpoint_{}.pth'.format(epoch))
        lr_scheduler.step(epoch)

if __name__ == '__main__':
    main()
