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
config.comments='exceed maximum states again no penalty.'
config.net=edict()  # ploci approximation networks'
config.train = edict() # training parameters
config.env = edict()
config.debug= edict()
config.env.max_states = 30  # maximum states in a single trial



config.outputPath = '../output'
config.timeStamp = time.asctime()
config.debug.debug =   True

config.train.optimizer = 'adam'
config.train.lr = .00001
config.train.wd = 1E-6
config.train.batch_size = 1000
config.train.ctx = mx.gpu() # by default
config.train.callback_batch_size = 3

config.train.trial = int(5E+4)
#config.train.exploration_trial = 1000
#config.train.exploration_th = .7
#config.train.exploration_mul = .9
#config.train.min_exploration = .1

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


