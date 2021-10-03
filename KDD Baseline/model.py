import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--x_data_path', '-x', default="./x_data_handcraft.npy")
parser.add_argument('--y_label_path', '-y', default="./y_label_handcraft.npy")
parser.add_argument('--uid_path','-u', default="./y_uid_handcraft.npy")
parser.add_argument('--train_set','-t', default='list/train_0.csv') 
parser.add_argument('--val_set','-v', default='list/val_0.csv') # not really needed, but keep it here for now
parser.add_argument('--model','-m',default='svm',help='choose from svm, rf, lr, and bt')
parser.add_argument('--data_part','-d',default='all',help='choose from all, breath, and cough')
args = parser.parse_args()

model_dict={'svm' : svm.SVC(class_weight='balanced'),
			'rf' : RandomForestClassifier(n_estimators=100, class_weight='balanced'),
			'lr' : LogisticRegression(class_weight='balanced'),
			'bt' : GradientBoostingClassifier()}
			

x_data = np.load(args.x_data_path)
y_data = np.load(args.y_label_path)
uid_data = np.load(args.uid_path)

train_set = pd.read_csv(args.train_set,header=None)

xtrain = [];
ytrain = [];
xval = [];
yval = [];

left_idx = 477 if args.data_part=='cough' else 0
right_idx = 477 if args.data_part=='breath' else 954

for x,y,uid in zip(x_data,y_data,uid_data):
	if uid in train_set.values:
		xtrain.append(x[left_idx:right_idx])
		ytrain.append(y)
	else:
		xval.append(x[left_idx:right_idx])
		yval.append(y)

clf= model_dict.get(args.model)

clf.fit(xtrain, ytrain)
ypred = clf.predict(xval)
print(args.train_set, args.model, "ROC_AUC:", str(metrics.roc_auc_score(yval, ypred)))