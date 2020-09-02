from __future__ import print_function

from cklib import ckstd
from cklib.ckstd import fprint
from cklib import ckmachine
import timeit
import numpy as np
from sklearn.metrics import f1_score
from cklib.cktime import date

class StaticThreshold:
    def __init__(self, logging = True):
        self.ppreds = None
        self.spreds = None
        self.pprobs = None
        self.y_true = None
        self.classes = None
        self.flows = None
        self.f1s = None
        self.isInit = False
        self.threshold = None
        self.th_range = [0., 1.]
        
        if logging:
            self.log = open('./log/' + date() + '.log', 'a')
        else:
            self.log = None
            
    def initializing(self, ppreds, pprobs, spreds, flows, classes, y_true, start = 0.5, end = 1.0):
        self.ppreds = ppreds
        self.pprobs = pprobs
        self.spreds = spreds
        self.flows = flows
        self.classes = classes
        self.y_true = y_true
        self.duration = duration
        self.protocol = protocol
        self.fin_count = fin_count
        self.ptime_mean = mean
        self.th_range[0] = start
        self.th_range[1] = end
        
        assert self.th_range[1] > self.th_range[0], 'ERROR: Threshold range is not collect'
        
        self.threshold = [round((a / 1000) + self.th_range[0], 3) for a in range(int((self.th_range[1] - self.th_range[0]) * 1000))]
        self.isInit = True
        fprint(self.log, 'Initializing compilte')
        
        return '<Initializing Threshold class>'
    
    def calculate(self):
        assert self.isInit, 'ERROR: Not initialized class'
        fprint(self.log, 'Start getting f1-scores')
        ts = timeit.default_timer()
        self.f1s = []
        for th in self.threshold:
            classified = []
            for flow_idx, fpreds, fprobs in zip(range(len(self.flows)), self.ppreds, self.spreds):
                found = False
                for pkt_idx, pred, prob in zip(range(len(fpreds)), fpreds, fprobs):
                    if th <= prob:
                        classified.append(pred)
                        found = True
                        break
                if not found:
                    classified.append(spreds[flow_idx])
            
            self.f1s.append(
                f1_score(
                    y_true = self.y_true,
                    y_pred = classified,
                    labels = self.classes,
                    average = 'macro'
                )
            )
        self.f1s = np.array(self.f1s)
        te = timeit.default_timer()
        fprint(self.log, '---> Done ({:.4f} sec)'.format(te - ts))
            
        return '<Function calculating f1-scores>'
    
    def getThreshold(self, omega = 1.0):
        assert self.f1s != None, 'ERROR: No f1-score data'
        idx = np.arange(len(self.f1s)[self.f1s >= omega][0])
        f1 = self.f1s[idx]
        th = self.threshold[idx]
        fprint(self.log, 'Omega: {} --> [F1-Score: {}] [Threshold: {}]'.format(omega, f1, th))
        return th
