"""
this file aims to CHAPTER17's Figure 17.1

Chen Y. Liang
Jan 14, 2018
"""

from config import config as cfg
import logging, time, os, copy
import mxnet as mx
import numpy as np
from mxnet import gluon
from rl_base import BASE_AGENT, BASE_ENV, train_agent, RL_metric



class chapter17_agt(BASE_AGENT):
    def __init__(self, batch_size, action_num):
        super(chapter17_agt, self).__init__(batch_size, action_num)
        self.max_state_exceed = False
        self.action_list = np.array([ [0,1],[1,0],[0,-1],[-1,0]]) # indexed by final_action, return coordinate increment
        self.init_state_list = [ [1,1], [2,1], [3,1], [4,1],\
                            [1,2], [3,2],\
                            [1,3], [2,3],[3,3]]
        self.dump_state_list = self.init_state_list

    def reset(self):
        super(chapter17_agt, self).reset()
        self.max_state_exceed = False
        idx = np.random.randint(0,len(self.init_state_list))
        #self.cur_state = np.array([1,1])  # reset coordinate
        self.cur_state = np.array(self.init_state_list[idx])  # reset coordinate

    def next_state(self):
        self.cur_state_data = mx.nd.reshape( mx.nd.array(self.cur_state), shape=(1,-1) )
        return self.cur_state_data

    def postForward(self):
        # append data
        #print('append into self.data_list')
        self.data_list.append( copy.deepcopy(self.cur_state_data) )
        self.trial_state_list.append(self.data_list[-1])

    def action(self, net_out):
        # exploration-exploitation
        """
            1. in chapter17( page646), thereis a transition model with probabilties: .8, .1, .1
            2. do exploration
        """
        net_out = net_out[0].asnumpy()
        decision_action = super(chapter17_agt, self).action(net_out) # sample an action
        r= np.random.uniform()
        if r < .8:# doit as it is
            self.final_action = decision_action
        elif r<.9: # left
            self.final_action = decision_action -1
        else:
            self.final_action = decision_action +1
        self.final_action = self.final_action % len(self.action_list)
        # track action & probability
        self.trial_action_list.append(self.final_action)
        self.act_prob_list.append( net_out[self.final_action] )
        # return the next coordinate
        return self.cur_state + self.action_list[self.final_action]






    def procFeedback(self,feedback):
        """
            manage ctl_state, step reward
            prepare for next_action
        """
        self.terminate_state = feedback['terminate_state']
        step_reward = feedback['step_reward']
        self.trial_reward_list.append(step_reward)
        if self.terminate_state is True: # handle final_reward
            self.latest_final_reward = feedback['final_reward']
            self.max_state_exceed = feedback['max_state_exceed']
        # prepare for next action
        collision = feedback['collision']
        if not collision: # change cur_state
            self.cur_state += self.action_list[self.final_action]
    def calc_dist(self):
        """
            call this every end of a trial
            to calculate out the label, grad scale for each step according2 the reward...

            reward() must return a scalar!

            return label, gradScale, all of list.
        """
        #self.final_reward()
        final_reward =self.latest_final_reward
        # by default, label = act
        self.label_list += self.trial_action_list[:]
        #trial_len = len(self.trial_action_list)
        # equal ditribution for final reward dot separate step reward
        """
            firstly, we implement textbook's example, use the whole reward as the only factor for decision making
        """
        tt_reward = sum(self.trial_reward_list)+ final_reward
        grad_mul = 0 if self.max_state_exceed else 1
        self.gradScale_list += [tt_reward * grad_mul for _ in self.trial_reward_list ]
        self.trial_reward_list, self.trial_action_list, self.trial_state_list = [], [], []
        #print('len: gradScale_list[%d], label_list[%d], data_list[%d]'%(len(self.gradScale_list), len(self.label_list), len(self.data_list)  ))

        return
        """
            sign-orinted decision
            as the second practice ...
        """
        self.gradScale_list += [ final_reward*step_reward for step_reward in self.trial_reward_list  ]


class chapter17_env(BASE_ENV):

    def __init__(self):
        self.wall = [[2,2],]
        self.postive_terminal_list= [[4,3],];self.postive_reward = 1
        self.negative_terminal_list =[[4,2],];self.negative_reward = -1
        self.still_reward = cfg.env.still_reward
        self.xlimits = [1, 4] # including boundaries
        self.ylimits = [1, 3]
        self.feedback = {}
        self.state_cnt = 0 # avoid infinte many trials

    def reset(self):
        self.state_cnt =0

    def Feedback(self, action):
        # action=(x,y) where the agent tends to be there
        """
        1. check boundaries
        2. check terminals
        """
        if self.xlimits[0] <= action[0] <= self.xlimits[1] and\
                self.ylimits[0] <= action[1] <= self.ylimits[1] and list(action) not in self.wall: # no collision
            self.feedback['collision'] = False
        else:
            self.feedback['collision'] = True

        if list(action) in self.postive_terminal_list:
            self.feedback['terminate_state'] = True
            self.feedback['final_reward'] = self.postive_reward
        elif list(action) in self.negative_terminal_list :
            self.feedback['terminate_state'] = True
            self.feedback['final_reward'] = self.negative_reward
        else: # step reward...
            self.feedback['terminate_state'] = False
            self.feedback['final_reward'] = None

        self.feedback['max_state_exceed'] = True if self.state_cnt > cfg.env.max_states else False

        self.feedback['step_reward']  = self.still_reward
        self.state_cnt += 1
        return self.feedback
logging.info(cfg)

agent = chapter17_agt(cfg.train.batch_size, 4)
env  = chapter17_env()

net= gluon.nn.HybridSequential()
net.add(\
        gluon.nn.BatchNorm(),
        gluon.nn.Dense(400, activation='relu'),\
        gluon.nn.BatchNorm(),\
        gluon.nn.Dense(500, activation='relu'),\
        gluon.nn.BatchNorm(),\
        gluon.nn.Dense(300, activation='relu'),\
        gluon.nn.Dense(4))
net.initialize(mx.init.Xavier(), ctx= cfg.train.ctx)
net.hybridize()

trainer = gluon.Trainer(net.collect_params(), cfg.train.optimizer,
                            {'learning_rate': cfg.train.lr, 'wd': cfg.train.wd})
#, 'momentum': opt.momentum},
metric = RL_metric(cfg.train.callback_batch_size, cfg.train.batch_size)

train_agent(env, agent, net, trainer, metric, batch_size= cfg.train.batch_size, ctx= cfg.train.ctx, trial=cfg.train.trial)


if not cfg.debug.debug:
    net.export(os.path.join(cfg.outputPath,cfg.timeStamp))
