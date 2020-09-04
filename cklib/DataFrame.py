'''
    Library: cklib.ckdata
    
    For management dataset
'''

# All imports
from __future__ import print_function

import itertools
import numpy as np
import timeit
import gc
import pandas as pd
import multiprocessing
import os, sys
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

import cklib
from cklib import ckconst
from cklib import cksess
from cklib.ckstd import fprint
from cklib.cktime import date
from cklib import ckstd

class Flow_Dataset:
    def __init__(self, logging = True, random_state = None):
        self.seed = random_state
        use_cores = multiprocessing.cpu_count() // 3 * 2
        
        self.train_dataset = None
        self.test_dataset = None
        self.train_session = None
        self.test_session = None
        self.train_flows = None
        self.test_flows = None
        
        self.pclf = RandomForestClassifier(n_jobs = use_cores, random_state = random_state)
        self.sclf = RandomForestClassifier(n_jobs = use_cores, random_state = random_state)
        self.le = LabelEncoder()
        self.pscaler = MinMaxScaler()
        self.sscaler = MinMaxScaler()
        
        self.spreds_train = None
        self.spreds_test = None
        self.sprobs_train = None
        self.sprobs_train_all = None
        self.sprobs_test = None
        self.sprobs_test_all = None
        
        self.ppreds_train = None
        self.ppreds_test = None
        self.pprobs_train = None
        self.pprobs_train_all = None
        self.pprobs_test = None
        self.pprobs_test_all = None

        self.train_ptime_mean = None
        self.train_stime_mean = None
        self.test_ptime_mean = None
        self.test_stime_mean = None
        
        self.train_durations = None
        self.test_durations = None
        self.train_fin_counts = None
        self.test_fin_counts = None
        self.train_protocols = None
        self.test_protocols = None
        
        if logging:
            self.log = open('./log/' + date() + '.log', 'a')
        else:
            self.log = None
        
    def read_csv(self, ptrain_path, ptest_path, strain_path, stest_path, encoding = ckconst.ISCX_DATASET_ENCODING):
        fprint(self.log, 'Reading dataset')
        ts = timeit.default_timer()
        
        self.train_dataset = pd.read_csv(filepath_or_buffer = ptrain_path, encoding = encoding).values
        self.test_dataset = pd.read_csv(filepath_or_buffer = ptest_path, encoding = encoding).values
        self.train_session = pd.read_csv(filepath_or_buffer = strain_path, encoding = encoding).values
        self.test_session = pd.read_csv(filepath_or_buffer = stest_path, encoding = encoding).values
        
        self.train_flows = cksess.get_flows(dataset = self.train_dataset)
        self.test_flows = cksess.get_flows(dataset = self.test_dataset)
        
        self.train_durations = self.train_dataset[:, 3]
        self.train_durations = [self.train_durations[flow] for flow in self.train_flows]
        self.test_durations = self.test_dataset[:, 3]
        self.test_durations = [self.test_durations[flow] for flow in self.test_flows]
        self.train_fin_counts = self.train_dataset[:, 45]
        self.train_fin_counts = [self.train_fin_counts[flow[-1]] for flow in self.train_flows]
        self.test_fin_counts = self.test_dataset[:, 45]
        self.test_fin_counts = [self.test_fin_counts[flow[-1]] for flow in self.test_flows]
        self.train_protocols = self.train_dataset[:, 2]
        self.train_protocols = [self.train_protocols[flow[-1]] for flow in self.train_flows]
        self.test_protocols = self.test_dataset[:, 2]
        self.test_protocols = [self.test_protocols[flow[-1]] for flow in self.test_flows]
        
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        return '<Function: read & shuffling csv>'
    
    def modelling(self):
        fprint(self.log, 'Training label encoder and scaler')
        ts = timeit.default_timer()
        self.le.fit(self.train_session[:, -1])
        self.le.fit(self.test_session[:, -1])
        self.pscaler.fit(self.train_dataset[:, 2:-1])
        self.sscaler.fit(self.train_session[:, 1:-1])
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Training random forest model')
        ts = timeit.default_timer()
        self.pclf.fit(
            X = self.pscaler.transform(self.train_dataset[:, 2:-1]),
            y = self.le.transform(self.train_dataset[:, -1])
        )
        
        self.sclf.fit(
            X = self.sscaler.transform(self.train_session[:, 1:-1]),
            y = self.le.transform(self.train_session[:, -1])
        )
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: modelling>'
    
    def predict(self):
        pred_ts = timeit.default_timer()
        
        fprint(self.log, 'Predict session training dataset')
        ts = timeit.default_timer()
        self.spreds_train = self.sclf.predict(self.sscaler.transform(self.train_session[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session training dataset predict time: {} seconds'.format(te - ts))

        self.train_stime_mean = (te - ts) / len(self.spreds_train)

        fprint(self.log, 'Predict session test dataset')
        ts = timeit.default_timer()
        self.spreds_test = self.sclf.predict(self.sscaler.transform(self.train_session[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session test dataset predict time: {} seconds'.format(te - ts))
        self.test_stime_mean = (te - ts) / len(self.spreds_test)

        self.sprobs_train_all = self.sclf.predict_proba(self.sscaler.transform(self.train_session[:, 1:-1]))
        self.sprobs_train = np.max(self.sprobs_train_all, axis = 1)
        self.sprobs_test_all = self.sclf.predict_proba(self.sscaler.transform(self.test_session[:, 1:-1]))
        self.sprobs_test = np.max(self.sprobs_test_all, axis = 1)
        
        fprint(self.log, 'Predict packet training dataset')
        ts = timeit.default_timer()
        self.ppreds_train = self.pclf.predict(self.pscaler.transform(self.train_dataset[:, 2:-1]))
        te = timeit.default_timer()
        self.train_ptime_mean = (te - ts) / len(self.ppreds_train)
        self.ppreds_train = [self.ppreds_train[flow] for flow in self.train_flows]
        self.pprobs_train_all = self.pclf.predict_proba(self.pscaler.transform(self.train_dataset[:, 2:-1]))
        self.pprobs_train = np.max(self.pprobs_train_all, axis = 1)
        self.pprobs_train_all = [self.pprobs_train_all[flow] for flow in self.train_flows]
        self.pprobs_train = [self.pprobs_train[flow] for flow in self.train_flows]
        fprint(self.log, 'Packet training dataset predict time: {} seconds'.format(te - ts))

        fprint(self.log, 'Predict packet test dataset')
        ts = timeit.default_timer()
        self.ppreds_test = self.pclf.predict(self.pscaler.transform(self.test_dataset[:, 2:-1]))
        te = timeit.default_timer()
        self.test_ptime_mean = (te - ts) / len(self.ppreds_test)
        self.ppreds_test = [self.ppreds_test[flow] for flow in self.test_flows]
        self.pprobs_test_all = self.pclf.predict_proba(self.pscaler.transform(self.test_dataset[:, 2:-1]))
        self.pprobs_test = np.max(self.pprobs_test_all, axis = 1)
        self.pprobs_test_all = [self.pprobs_test_all[flow] for flow in self.test_flows]
        self.pprobs_test = [self.pprobs_test[flow] for flow in self.test_flows]
        fprint(self.log, 'Packet test dataset predict time: {} seconds'.format(te - ts))

        pred_te = timeit.default_timer()
        fprint(self.log, 'Processing of predict part is finished ({} seconds)'.format(pred_te - pred_ts))
        
        return '<Function: predict>'
    
    def getLabelEncoder(self):
        return self.le
    def getClassifier(self):
        return self.pclf, self.sclf
    
    def getTrainPred(self):
        return self.ppreds_train, self.spreds_train
    def getTrainProb(self):
        return self.pprobs_train, self.sprobs_train
    def getTrainPktProb(self):
        return self.pprobs_train_all
    def getTrainFlow(self):
        return self.train_flows
    def getTrainLabel(self):
        return self.train_session[:, -1]
    def getTrainMean(self):
        return self.train_ptime_mean, self.train_stime_mean
    def getTrainDuration(self):
        return self.train_durations
    def getTrainFin(self):
        return self.train_fin_counts
    def getTrainProtocol(self):
        return self.train_protocols
    
    def getTestPred(self):
        return self.ppreds_test, self.spreds_test
    def getTestProb(self):
        return self.pprobs_test, self.sprobs_test
    def getTestPktProb(self):
        return self.pprobs_test_all
    def getTestFlow(self):
        return self.test_flows
    def getTestLabel(self):
        return self.test_session[:, -1]
    def getTestMean(self):
        return self.test_ptime_mean, self.test_stime_mean
    def getTestDuration(self):
        return self.test_durations
    def getTestFin(self):
        return self.test_fin_counts
    def getTestProtocol(self):
        return self.test_protocols
    
class Session_Dataset:
    def __init__(self, split = 0.7, clf = 'rf', logging = True, random_state = None):
        self.train_dataset = None
        self.test_dataset = None

        self.split_ratio = split
        self.train_size = None
        self.seed = random_state
        self.skip_datas = []
        
        np.random.seed(random_state)
        use_cores = multiprocessing.cpu_count() // 4 * 3
        if clf == 'rf':
            self.sclf = RandomForestClassifier(n_jobs = use_cores, random_state = random_state)
        elif clf == 'dt':
            self.sclf = DecisionTreeClassifier(random_state = random_state)
        elif clf == 'et':
            self.sclf = ExtraTreeClassifier(random_state = random_state)
        elif clf == 'adt':
            self.sclf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = random_state), random_state = random_state)
        elif clf == 'arf':
            self.sclf = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_jobs = use_cores, random_state = random_state), random_state = random_state)
        elif clf == 'gbt':
            self.sclf = GradientBoostingClassifier(random_state = random_state)
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        
        self.spreds_train = None
        self.spreds_test = None

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
        dataset = pd.read_csv(filepath_or_buffer = path, encoding = encoding).values
        
        fprint(self.log, 'Skip data: {}'.format(self.skip_datas))
        for word in self.skip_datas:
            dataset = dataset[dataset[:, -1] != word]
            
        self.train_size = int(len(dataset) * self.split_ratio)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Shuffling dataset by flows')
        ts = timeit.default_timer()
        np.random.shuffle(dataset)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Creating training & test dataset')
        ts = timeit.default_timer()
        self.train_dataset = dataset[:self.train_size]
        self.test_dataset = dataset[self.train_size:]
        gc.collect()
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        return '<Function: read & shuffling csv>'

    def modelling(self):
        fprint(self.log, 'Training label encoder and scaler')
        ts = timeit.default_timer()
        self.le.fit(self.train_dataset[:, -1])
        self.le.fit(self.test_dataset[:, -1])
        self.scaler.fit(self.train_dataset[:, 1:-1])
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)\n'.format(te - ts))
        
        fprint(self.log, 'Training model')
        ts = timeit.default_timer()

        self.sclf.fit(
            X = self.scaler.transform(self.train_dataset[:, 1:-1]),
            y = self.le.transform(self.train_dataset[:, -1])
        )
        
        gc.collect()
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: modelling>'

    def predict(self):
        pred_ts = timeit.default_timer()
        
        fprint(self.log, 'Predict session training dataset')
        ts = timeit.default_timer()
        self.spreds_train = self.sclf.predict(self.scaler.transform(self.train_dataset[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session training dataset predict time: {} seconds'.format(te - ts))

        fprint(self.log, 'Predict session test dataset')
        ts = timeit.default_timer()
        self.spreds_test = self.sclf.predict(self.scaler.transform(self.test_dataset[:, 1:-1]))
        te = timeit.default_timer()
        fprint(self.log, 'Session test dataset predict time: {} seconds'.format(te - ts))

        return '<Function: predict>'

    def getTrainPredict(self):
        return self.spreds_train

    def getTestPredict(self):
        return self.spreds_test

    def getLabelEncoder(self):
        return self.le

    def getTrainLabel(self):
        return self.train_dataset[:, -1]

    def getTestLabel(self):
        return self.test_dataset[:, -1]

    def getClassifier(self):
        return self.sclf
    
class SessionConverter:
    def __init__(self, logging = True):
        self.dataset = None
        self.session = None
        self.header = None
        self.isSess = False
        
        if logging:
            self.log = open('./log/' + date() + '.log', 'a')
        else:
            self.log = None
        
    def read_csv(self, path, encoding = ckconst.ISCX_DATASET_ENCODING):
        fprint(self.log, 'Read dataset: {}'.format(path))
        ts = timeit.default_timer()
        self.dataset = pd.read_csv(filepath_or_buffer = path, encoding = encoding)
        self.header = self.dataset.columns.tolist()
        self.dataset = self.dataset.values
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: read csv>'
    
    def sessionization(self):
        fprint(self.log, 'Convert packet dataset to session dataset')
        ts = timeit.default_timer()
        flows = cksess.get_flows(self.dataset)
        self.session = self.dataset[[flow[-1] for flow in flows]]
        self.isSess = True
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        
        return '<Function: sessionization>'
    
    def getSessionDataset(self):
        return self.session
    
    def save(self, path):
        if self.isSess:
            fprint(self.log, 'Writing session dataset at {}'.format(path))
            ts = timeit.default_timer()
            pd.DataFrame(data = self.session).to_csv(path, index = False, header = self.header, encoding = ckconst.ISCX_DATASET_ENCODING)
            te = timeit.default_timer()
            fprint(self.log, '---> Done ({:.4f} seconds)'.format(te - ts))
        else:
            return 'ERROR: Not sessionization'
        return '<Function: Save session>'
