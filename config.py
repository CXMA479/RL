"""
time for configuration

Chen Y. Liang
Jan 14, 2018
"""

from easydict import EasyDict as edict
import logging
import time, os
import numpy as np
import mxnet as mx

config = edict()
config.net=edict()  # ploci approximation networks'
config.train = edict() # training parameters
config.env = edict()
config.debug= edict()
config.env.max_states = 30  # maximum states in a single trial



config.outputPath = '../output'
config.timeStamp = time.asctime()
config.debug.debug = not True

config.train.optimizer = 'adam'
config.train.lr = .001
config.train.wd = 1E-6
config.train.batch_size = 2000
config.train.ctx = mx.gpu() # by default
config.train.callback_batch_size = 90

config.train.trial = int(2E+6)
config.train.exploration_trial = 20000
config.train.exploration_th = .5
config.train.exploration_mul = .8
config.train.min_exploration = .1

config.env.still_reward = -0.04





LOGFMT='%(levelname)s: %(asctime)s %(pathname)s [line: %(lineno)d]  %(message)s'
#LOGFMT='%(levelname)s: %(asctime)s %(pathname)s %(filename)s [line: %(lineno)d]  %(message)s'
filePathAndNname = os.path.join(config.outputPath,config.timeStamp+'.log')
if config.debug.debug:
  logging.basicConfig(level=logging.DEBUG,format=LOGFMT)
elif os.path.isdir(config.outputPath) :

  logging.basicConfig(level=logging.INFO,
            filename=filePathAndNname,
                    format=LOGFMT
            )
  console = logging.StreamHandler()
  logging.getLogger('').addHandler(console)
  formatter = logging.Formatter(LOGFMT)
  console.setFormatter(formatter)


def nofile():
    os.remove(filePathAndNname)
    logging.basicConfig(level=logging.INFO,format=LOGFMT)


