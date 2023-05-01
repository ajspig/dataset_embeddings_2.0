#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:16:56 2020

@author: hsjomaa
"""

from dataset import Dataset
from sampling import Batch,Sampling,TestSampling
import tensorflow as tf
import copy
import json
from model import Model
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import os
import pdb

# set random seeds
tf.random.set_seed(0)
np.random.seed(42)

# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--configuration', help='Select model configuration', type=int,default=0)
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--LookAhead', help='Implement lookahead on backend optimizer', type=str,choices=['True','False'],default='False')
parser.add_argument('--searchspace', help='Select metadataset',choices=['a','b','c'], type=str,default='a')
parser.add_argument('--learning_rate', help='Learning rate',type=float,default=1e-3)
parser.add_argument('--delta', help='negative datasets weight',type=float,default=2)
parser.add_argument('--gamma', help='distance hyperparameter',type=float,default=1)

args    = parser.parse_args() #Namespace(configuration=0, split=0, LookAhead='False', searchspace='a', learning_rate=0.001, delta=2, gamma=1)

# pdb.set_trace()
rootdir     = os.path.dirname(os.path.realpath(__file__)) #/users/guest/j/jhiggin6/Documents/Thesis/d2v
config_file = os.path.join(rootdir, "configurations",f"configuration-{args.configuration}.json") #/users/guest/j/jhiggin6/Documents/Thesis/d2v/configurations/configuration-0.json
info_file   = os.path.join(rootdir, "metadataset"  ,"info.json") #/users/guest/j/jhiggin6/Documents/Thesis/d2v/metadataset/info.json

configuration = json.load(open(config_file,'r'))
### added by Josh ###
#  configuration = {
#  'nonlinearity_d2v': 'relu', 'units_f': 32, 'nhidden_f': 4, 'architecture_f': 'SQU',
#  'resblocks_f': 8, 'units_h': 32, 'nhidden_h': 4, 'architecture_h': 'SQU', 
#  'resblocks_h': 8, 'units_g': 32, 'nhidden_g': 4, 'architecture_g': 'SQU', 
#  'ninstanc': 256, 'nclasses': 5, 'nfeature': 32, 'number': 0
#  }
### end added by Josh ###
# update with shared configurations with specifics
config_specs = {
    'split':	args.split,
    'LookAhead':	eval(args.LookAhead),
    'searchspace':	args.searchspace,
    'learning_rate':	args.learning_rate,
    'delta':	args.delta,
    'gamma':	args.gamma,
    'minmax':	True,    # not in args
    'batch_size':	16,  # not in args
    }
configuration.update(config_specs)

searchspaceinfo = json.load(open(info_file,'r'))
configuration.update(searchspaceinfo[args.searchspace])
### added by Josh ###
#  configuration = {
#  'nonlinearity_d2v': 'relu', 'units_f': 32, 'nhidden_f': 4, 'architecture_f': 'SQU',
#  'resblocks_f': 8, 'units_h': 32, 'nhidden_h': 4, 'architecture_h': 'SQU',
#  'resblocks_h': 8, 'units_g': 32, 'nhidden_g': 4, 'architecture_g': 'SQU',
#  'ninstanc': 256, 'nclasses': 3, 'nfeature': 16, 'number': 0,
#  'split': 0, 'LookAhead': False, 'searchspace': 'a', 'learning_rate': 0.001,
#  'delta': 2, 'gamma': 1, 'minmax': True, 'batch_size': 16, 'cardinality': 256,
#  'D': 10
#  }

# pdb.set_trace()
### end added by Josh ###
# create Dataset
normalized_dataset = Dataset(configuration,rootdir,use_valid=True)

# load training sets
nsource = len(normalized_dataset.orig_data['train'])
ntarget = len(normalized_dataset.orig_data['valid'])
ntest   = len(normalized_dataset.orig_data['test'])
# learning rate scheduler
optimizer        = tf.keras.optimizers.Adam(configuration['learning_rate'])
# create training model
# create model
model     = Model(configuration,rootdir=rootdir)
batch     = Batch(configuration['batch_size'])
# create evaluation model (for validation)

testconfiguration = copy.copy(configuration)
testconfiguration['batch_size'] = 16 if args.searchspace != 'c' else 18

testmodel   = Model(testconfiguration,rootdir=rootdir,for_eval=True)
testbatch   = Batch(testconfiguration['batch_size'])

# define list/csv tracking
print(model.model.summary())

# Define training parameters
epochs = 10000
# reset metric trackers
model.reset_states()    

# Start training56
sampler     = Sampling(dataset=normalized_dataset)
testsampler = TestSampling(dataset=normalized_dataset)

early = 0
best_auc = -np.inf
model.reset_states()  
for epoch in range(epochs):
    batch = sampler.sample(batch,split='train',sourcesplit='train')
    batch.collect()
    metrics = model.train_step(x=batch.input,y=batch.output,optimizer=optimizer,clip=True)
    if optimizer.iterations%50==0:
        iteration = optimizer.iterations.numpy()
        # save as current weights
        model.save_weights(iteration=optimizer.iterations.numpy())
        # reset evaluation mse tracker
        savedir       = os.path.join(model.directory,f"iteration-{iteration}")
        y_true = []
        y_pred = []
        for _ in range(ntarget):
            reuse = False
            for q in range(10):
                batch = testsampler.sample(batch,split='valid',sourcesplit='train',targetdataset=_)
                reuse = True
                batch.collect()
                prob,label = model.predict(batch.input,batch.output)
                y_pred.append(prob)
                y_true.append(label)
        auc_score = roc_auc_score(np.hstack(y_true),np.hstack(y_pred))
        model._update_metrics(metrics={"roc":auc_score})
        model.update_tracker(training=True,metrics=model.metrics)
        model.report()
        if np.abs(model.metrics['roc'].result().numpy()-best_auc)>1e-3:
            best_auc = auc_score
            early = 0
        else:
            early +=1
    model.dump()
    sampler_file = os.path.join(model.directory,"distribution.csv")
    sampler.distribution.to_csv(sampler_file)
    testsampler.distribution.to_csv(os.path.join(model.directory,"valid-distribution.csv"))
    if early > 16:
        break

model.save_weights(iteration=optimizer.iterations.numpy())   
import pandas as pd
metafeatures = pd.DataFrame(data=None)
splitmf = []
filesmf = []
for splits in [("train",nsource),("valid",ntarget),("test",ntest)]:
    for _ in range(splits[1]):
        datasetmf = []
        reuse = False
        for q in range(10):
            batch = testsampler.sample(batch,split=splits[0],sourcesplit='train',targetdataset=_)
            reuse = True
            batch.collect()
            datasetmf.append(model.getmetafeatures(batch.input))
        splitmf.append(np.vstack(datasetmf).mean(axis=0)[None])
    filesmf +=normalized_dataset.orig_files[splits[0]]
splitmf = np.vstack(splitmf)
metafeatures = pd.DataFrame(data=splitmf,index=filesmf)
metafeatures.to_csv(os.path.join(model.directory,"meta-feautures.csv"))