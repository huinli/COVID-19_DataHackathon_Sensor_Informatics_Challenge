import os
import argparse
parser = argparse.ArgumentParser(description='Train autospeech network')
# general
parser.add_argument('--d',
                    help='experiment configure file name',
                    required=True,
                    type=str)
args = parser.parse_args()

if not os.path.exists('saves/{}'.format(args.d)):
    os.mkdir('saves/{}'.format(args.d))
if os.path.exists('logs'):
    os.system("mv  logs saves/{}/logs".format(args.d))
if os.path.exists('logs_search'):
    os.system("mv logs_search saves/{}/logs_search".format(args.d))

if os.path.exists('logs_scratch'):
    os.system("mv logs_scratch saves/{}/logs_scratch".format(args.d))

if os.path.exists('data'):
    os.system("mv data saves/{}/data".format(args.d))
