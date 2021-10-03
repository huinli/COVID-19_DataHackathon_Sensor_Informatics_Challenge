from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import resnet
from config import cfg, update_config
from utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset_ori_test import DeepSpeakerDataset
from functions import train_from_scratch, validate_identification, validate_identification_balance,validate_identification_roc
from loss import CrossEntropyLoss
import sys

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

def getModel(args,cfg,path,model = None):
    if model == None:
        model = eval('resnet.{}(num_classes={})'.format(cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()
    #checkpoint
    checkpoint = os.path.join(args.load_path, 'checkpoint','best_models',path)
    assert os.path.exists(checkpoint)
    _checkpoint = torch.load(checkpoint)
    model.load_state_dict(_checkpoint['state_dict'])
    return model

def evalate(load_path):
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

    #s_path = cfg.DATASET.DATA_DIR + '/speech_{}_data'.format(proc_num)
    s_path = cfg.DATASET.DATA_DIR + '/saved_data/speech_test_data'
    
    test_dataset_identification = DeepSpeakerDataset(
        Path(s_path),'test', cfg.DATASET.PARTIAL_N_FRAMES, None, is_test=False)


    test_loader_identification = torch.utils.data.DataLoader(
        dataset=test_dataset_identification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    model = getModel(args,cfg,'speech_{}.pth'.format(proc_num))   
    p = validate_identification(cfg,model,test_loader_identification,proc_num)
if __name__ == "__main__":
    evalate(sys.argv[1])
