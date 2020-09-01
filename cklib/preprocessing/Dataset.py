'''
    Library: cklib.Dataset
    
    For management dataset
'''

# All imports
from __future__ import print_function

import numpy as np
import timeit
import pandas as pd
import os, sys
import joblib

import cklib
from cklib import ckconst
from cklib import cksess
from cklib.ckstd import fprint
from cklib.cktime import date
from cklib import ckstd

class PacketToSession:
    def __init__(self):
        pass
        
class PacketSplit:
    def __init__(self, split_ratio = 0.7, logging = True, random_state = None):
        self.origin_data = None
        self.train_data = None
        self.test_data = None
        self.header = None
        self.split_ratio = split_ratio
        self.seed = random_state
        self.skip_datas = []
        
        if logging:
            self.log = open('./log/' + date() + '.log', 'a')
        else:
            self.log = None
        
    def skip_data(self, *skip):
        for word in skip:
            self.skip_datas.append(word)
        return '<Function: skip data>'
    
    def read_csv(self, path, encoding = ckconst.ISCX_DATASET_ENCODING):
        fprint(self.log, 'Reading dataset: {}'.format(path))
        ts = timeit.default_timer()
        dataset = pd.read_csv(filepath_or_buffer = path, encoding = encoding)
        self.header = dataset.columns.tolist()
        dataset = dataset.values
        
        fprint(self.log, 'Skip data: {}'.format(self.skip_datas))
        for word in self.skip_datas:
            dataset = dataset[dataset[:, -1] != word]
            
        flows = cksess.get_flows(dataset = dataset)
        train_size = int(len(flows) * self.split_ratio)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Shuffling dataset by flows')
        ts = timeit.default_timer()
        dataset, _ = cksess.shuffle_flow(dataset = dataset, flows = flows, random_state = self.seed)
        flows = cksess.get_flows(dataset = dataset)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Creating training & test dataset')
        ts = timeit.default_timer()
        session = dataset[[flow[-1] for flow in flows]]
        self.train_session = session[:train_size]
        self.test_session = session[train_size:]
        self.train_dataset = dataset[cksess.flatten(flows[:train_size])]
        self.test_dataset = dataset[cksess.flatten(flows[train_size:])]
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        return '<Function: read & shuffling csv>'
    
    def save(self, master_path, name):
        fprint(self.log, 'Writing session dataset at {}'.format(master_path))
        path = [
            master_path + name + '_pkt_train.csv',
            master_path + name + '_pkt_test.csv',
            master_path + name + '_sess_train.csv',
            master_path + name + '_sess_test.csv'
        ]
        
        ts = timeit.default_timer()
        pd.DataFrame(data = self.train_dataset).to_csv(path[0], index = False, header = self.header, encoding = ckconst.ISCX_DATASET_ENCODING)
        pd.DataFrame(data = self.test_dataset).to_csv(path[1], index = False, header = self.header, encoding = ckconst.ISCX_DATASET_ENCODING)
        pd.DataFrame(data = self.train_session).to_csv(path[2], index = False, header = self.header, encoding = ckconst.ISCX_DATASET_ENCODING)
        pd.DataFrame(data = self.test_session).to_csv(path[3], index = False, header = self.header, encoding = ckconst.ISCX_DATASET_ENCODING)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: save dataset>'
    