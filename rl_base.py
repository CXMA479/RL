"""
this work tries to construct a framework for POLICY GRADIENT training

Chen Y. Liang
4 Jan, 2017
"""

import mxnet as mx
import time, logging
import numpy as np
import matplotlib.pyplot as plt
import os, sys, copy
from mxnet import autograd


class BASE_ENV(object):
    """
        responses to the agent's action:
    """
    def __init__(self):
        pass

    def Feedback(self, action):
        """
            accepts actions from agent as the key_args
        """
        assert 0, 'not defined'
    def reset(self):
        raise NotImplementedError



class BASE_AGENT(object):
    """
    manages the env's FEEDBACK( including reward, sign), seqs of the taken ACTIONs & experienced STATEs
    """
    def __init__(self, batch_size, action_num):
        self.batch_size = batch_size
        self.action_num = action_num
        self.cur_trial = -1 # self.reset step this value
        self.latest_final_reward=None
        self.cur_state = None
        self.ctl_dict = {} # for controll
        self.terminate_state = False # sign for end this trial

        self.trial_action_list = []  # final action track
        self.trial_reward_list = [] # EVERY step may have immediate reward, by default, 0
        self.trial_state_list =[]

        self.gradScale_list, self.label_list = [], [] # stacked contents
        self.data_list, self.final_reward_list = [], []
        self.act_prob_list =[] # track action probability reference2 Monte Carlo Policy Gradient

        self.feedback = None   # from environment

        pass

    def procFeedback(self, feedback):
        """
            feedback from env as key_args
            task list:
                1. append trial_reward_list
                2. accumulate account for the final reward's calculation
                3. prepare for next state
        """
        raise NotImplementedError()

    def postForward(self):
        """
            track cur_state.DataBatch.data( by default), or ...
        """
        self.data_list.append( copy.deepcopy(self.cur_state) )
#        self.trial_data_list.append(self.stacked_DataBatch_list_dict['data'][-1]) # just a ref

    def final_reward(self):
        """
            invoked every end of a trial, usually from calc_dist()
        """
        self.latest_final_reward = None
        self.final_reward_list.append(self.latest_final_reward)
        raise NotImplementedError

    def calc_dist(self):
        """
            call this every end of a trial
            to calculate out the label, grad scale for each step according2 the reward...

            reward() must return a scalar!

            return label, gradScale, all of list.
        """
        #self.final_reward()
        final_reward =self.latest_final_reward
        #trial_len = len(self.trial_action_list)
        # equal ditribution for final reward dot separate step reward
        self.gradScale_list += [ final_reward*step_reward for step_reward in self.trial_reward_list  ]
        # by default, label = act
        self.label_list += self.trial_action_list[:]

    def next_state(self):
        """
            update cur_state
            type(ret) = mx.io.DataBatch for net
        """
        raise NotImplementedError

    def action(self, net_out):
        """
            take exploration-exploitation
            append a scalar to the trial_action_list

            sample from the net_output to get the final action as a BASIC operation
            net_output:
                output by softmax
        """
        N=20
        while True:
            act_list = np.random.randint(0, self.action_num, (N,))
            try:
                idx = list(net_out[act_list] > np.random.uniform(size=(N,))).index(True)
#            assert 0, idx
                act = act_list[idx]
            except:
                continue
#            logging.debug('return from sampling')
            return act


        #raise NotImplementedError()

    def reset(self):
        self.terminate_state = False
        self.cur_trial  += 1

    def DataBatch_gradScale_batch(self):
        """
            i think i can do something for inheriting
        """
        #assert 0, ( len(self.gradScale_list), len(self.data_list), len(self.label_list) )
        assert len(self.gradScale_list) == len(self.data_list) \
                and len(self.gradScale_list) >= self.batch_size,'len gradScale_list[%d] vs len data_list[%d]'%(len(self.gradScale_list), len(self.data_list) )+'\tlen of the list[%d] vs batch_size[%d]'%(\
                    len(self.data_list), self.batch_size)
        data_shape = self.data_list[0].shape[1:]
        Data = mx.nd.empty( (self.batch_size,)+ data_shape, dtype=np.float32 )
        label = mx.nd.empty( (self.batch_size, ), dtype=np.float32 )
        outgrad = mx.nd.empty( (self.batch_size, ) , dtype = np.float32 )
        batch_passed = 0
#        print('begin gen data for forward...')
        while batch_passed < self.batch_size:
#            print( len(self.data_list) )
            Data[batch_passed] = mx.nd.reshape( self.data_list.pop(0), shape= data_shape )
            label[batch_passed]  = self.label_list.pop(0)
            outgrad[batch_passed] = self.gradScale_list.pop(0)/(0*self.act_prob_list.pop(0) +1 )
            batch_passed  += 1
        return Data, label, outgrad

class RL_metric():
    def __init__(self, batch_freq, batch_size):
        self.final_reward_list=[] # cause, usually, the trial's length is unpredictable 
        self.batch_passed = 0
        self.batch_freq = batch_freq
        self.batch_size = batch_size
        self.t0, self.t1= time.time(), 0.
    def update(self, final_reward):
        self.final_reward_list.append(final_reward)
    def check_batch(self):
        #self.batch_passed += 1
        if self.batch_passed% self.batch_freq==0 and self.batch_passed is not 0: # info the metric...
            self.t1=time.time()
            time_elapsed = self.t1- self.t0
            logging.info('trials[%d]\tbatch[%d]\tReward: mean[%.6f], var[%.6f]\t%.2f states/s'%(\
                        len(self.final_reward_list), self.batch_passed, np.mean(self.final_reward_list), np.var(self.final_reward_list), self.batch_size*self.batch_freq/ time_elapsed ) )
            self.final_reward_list = []
            self.t0 = self.t1
        self.batch_passed += 1


def train_agent(env, agent,  net, trainer, metric,  batch_size, trial, ctx):

    """
        env, agent:
            all instantiated objects
        net:
            approximating mx.mod.Module
    """
    final_reward_list=[]
    Loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    for trial_i in xrange(trial):
        agent.reset()
        env.reset()
        while agent.terminate_state is not True:
            state = agent.next_state()
            state = state.as_in_context(ctx)
            net_out = net(state) # policy approximation
            net_out  = mx.nd.softmax(net_out)
            agent.postForward()  # do something e.g. track DataBatch
            action = agent.action( net_out )      # exploration-exploitation
            feedback = env.Feedback(action)             # response from env
            agent.procFeedback(feedback)        # prepare for next state require, check if terminated

        if len( metric.final_reward_list ) == 0:
            logging.info('sample state:'+str( [ list(c.asnumpy().astype(int)[0]) for c in agent.trial_state_list+ [agent.next_state()]] ))
        agent.calc_dist()  # according to the reward & states
        metric.update( agent.latest_final_reward )

#        print('agent.label_list len[%d], batch_size[%d]'%( len(agent.label_list), batch_size) )
        if len(agent.label_list) < batch_size: # skip forward_backward
            continue
        while len(agent.label_list) > batch_size: # dry up the pool for mem space
            Data, label, outgrad = agent.DataBatch_gradScale_batch()
            Data, label, outgrad =[ _.as_in_context(ctx) for _ in [Data, label, outgrad] ]
            #assert 0, (Data.shape, label.shape, outgrad.shape)
            with autograd.record():
                y =net(Data)
                L = Loss(y, label)
                L.backward(outgrad)
#                print('backward ends.')

#            logging.info('test logging.')
#            mx.nd.waitall()
            trainer.step(Data.shape[0])
#            print('step ends.')

            metric.check_batch()





