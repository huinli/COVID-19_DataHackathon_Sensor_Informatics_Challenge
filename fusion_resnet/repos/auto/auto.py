import os
import argparse
parser = argparse.ArgumentParser(description='Train autospeech network')
# general
parser.add_argument('--d',
                    help='experiment configure file name',
                    required=True,
                    type=str)
args = parser.parse_args()
os.system('cp -R /home/zhengxio/Xingyu/VoiceEntropyV2/dataset/generated/voxcelebs/{} data'.format(args.d))

os.system('python data_preprocess.py data')
# os.system(
#     'python train_baseline_identification.py --cfg exps/baseline/resnet18_iden.yaml')
# os.system(
#     'python train_baseline_identification.py --cfg exps/baseline/resnet34_iden.yaml')

os.system('python search.py --cfg exps/search.yaml')
