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
from models import ffn as fn
from config import cfg, update_config
from utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset_test import DeepSpeakerDataset
from functions import train_from_scratch, test_identificate
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

def getmodels(args,cfg,path,model = None):
    if model == None:
        model = eval('resnet.{}(num_classes={})'.format(cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()
    #checkpoint
    checkpoint = os.path.join(args.load_path, 'checkpoint','best_models',path)
    assert os.path.exists(checkpoint)
    _checkpoint = torch.load(checkpoint)
    model.load_state_dict(_checkpoint['state_dict'])
    return model

def getData(cfg,path1,path2,path3):
    test_dataset_identification = DeepSpeakerDataset(
        Path(path1),Path(path2),Path(path3),'test', cfg.DATASET.PARTIAL_N_FRAMES, None, is_test=False)
    print(cfg.TRAIN.BATCH_SIZE)
    
    test_loader_identification = torch.utils.data.DataLoader(
        dataset=test_dataset_identification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    return test_loader_identification

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
    
    # model and optimizer
    model_breath = getmodels(args,cfg,'breathing_{}.pth'.format(proc_num))
    model_cough = getmodels(args,cfg,'cough_{}.pth'.format(proc_num))
    model_speech = getmodels(args,cfg,'speech_{}.pth'.format(proc_num))
    ffn_model = fn.ffn()
    ffn = getmodels(args,cfg,'ffn_{}.pth'.format(proc_num), ffn_model)    
    b_path = cfg.DATASET.DATA_DIR +'/saved_data/breathing_test_data'
    c_path = cfg.DATASET.DATA_DIR + '/saved_data/cough_test_data'
    s_path = cfg.DATASET.DATA_DIR + '/saved_data/speech_test_data'
    test_loader_identification = getData(cfg,b_path,c_path,s_path)
    
    criterion = CrossEntropyLoss(2).cuda()
    map1 = test_identificate(cfg, model_breath,model_cough,model_speech,ffn, test_loader_identification, criterion)
    #with open("roc.txt","a") as f:
        #f.write("{} {} {} \n".format(load_path,roc,acc))
if __name__ == "__main__":
    evalate(sys.argv[1])
