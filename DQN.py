# this model is based on the code from https://github.com/AndersonJo/dqn-pytorch/blob/master/dqn.py

import argparse
import copy
import glob
import logging
import math
import os
import re
from collections import deque
from collections import namedtuple
from random import random, sample

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from DQN_env import Environment
from DQN_network import DQN, LSTMDQN

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS = 50
TRAIN_SIZE = 50_000

# Training
BATCH_SIZE = 64

# Replay Memory
REPLAY_MEMORY = 10000  # was 50000 consider changing

# Epsilon - exploratory behavior
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10_000_000

# LSTM Memory
LSTM_MEMORY = 128

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")  # device to run the model on

# ETC Options
TARGET_UPDATE_INTERVAL = 1000
CHECKPOINT_INTERVAL = 5000
PLAY_INTERVAL = 5000  # basically when to evaluate
PLAY_REPEAT = 1000  # how many times to evaluate
LEARNING_RATE = 0.01

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('--model', default='dqn', type=str, help='forcefully set step')
parser.add_argument('--step', default=None, type=int, help='forcefully set step')
parser.add_argument('--best', default=None, type=int, help='forcefully set best')
parser.add_argument('--load_latest', dest='load_latest', action='store_true', help='load latest checkpoint')
parser.add_argument('--no_load_latest', dest='load_latest', action='store_false', help='train from the scrach')
parser.add_argument('--checkpoint', default=None, type=str, help='specify the checkpoint file name')
parser.add_argument('--mode', dest='mode', default='train', type=str, help='[play, train, inspect]')
parser.add_argument('--clip', dest='clip', action='store_true', help='clipping the delta between -1 and 1')
parser.add_argument('--noclip', dest='clip', action='store_false', help='not clipping the delta')
parser.add_argument('--skip_action', default=1, type=int, help='Skipping actions')
parser.add_argument('--inspect', dest='inspect', action='store_true', help='Inspect CNN')
parser.add_argument('--seed', default=111, type=int, help='random seed')
parser.set_defaults(clip=True, load_latest=True, inspect=False)
parser: argparse.Namespace = parser.parse_args()

# Random Seed
torch.manual_seed(parser.seed)
torch.cuda.manual_seed(parser.seed)
np.random.seed(parser.seed)

# Logging
logger = logging.getLogger('DQN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')

file_handler = logging.FileHandler(f'dqn_{parser.model}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class ReplayMemory(object):
    def __init__(self, capacity=REPLAY_MEMORY):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self._available = False

    def put(self, state: np.array, action: torch.LongTensor, reward: np.array, next_state: np.array):  # why arrays?
        state = (torch.FloatTensor(state[0, 0]), torch.FloatTensor(state[0, 1]))
        reward = torch.FloatTensor([reward])
        if next_state is not None:
            next_state = (torch.FloatTensor(next_state[0, 0]), torch.FloatTensor(next_state[0, 1]))
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return self.Transition(*(zip(*transitions)))

    def size(self):
        return len(self.memory)

    def is_available(self):
        if self._available:
            return True

        if len(self.memory) > BATCH_SIZE:
            self._available = True
        return self._available


class Agent(object):

    def __init__(self, args: argparse.Namespace, cuda=False, action_repeat: int = 1):
        # Init
        self.clip: bool = args.clip
        self.seed: int = args.seed
        self.action_repeat: int = action_repeat
        self.frame_skipping: int = args.skip_action
        self._state_buffer = deque(maxlen=self.action_repeat)
        self.step = 0
        self.best_score = args.best or -10000
        self.best_count = 0

        # Environment
        self.env = Environment(seed=self.seed, block_size=2, use_cuda=True)

        # DQN Model
        self.dqn_hidden_state = self.dqn_cell_state = None
        self.target_hidden_state = self.target_cell_state = None

        self.mode: str = args.model.lower()
        if self.mode == 'dqn':
            self.dqn: DQN = DQN(self.env.action_space).to(device)
        elif self.mode == 'lstm':
            self.dqn: LSTMDQN = LSTMDQN(self.env.action_space)

            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.dqn.init_states()
            self.target_hidden_state, self.target_cell_state = self.dqn.init_states()

            # For Training Play
            self.train_hidden_state, self.train_cell_state = self.dqn.init_states()

            # For Validation Play
            self.test_hidden_state, self.test_cell_state = self.dqn.init_states()

        if use_cuda:
            self.dqn.cuda()

        # DQN Target Model
        self.target: DQN = copy.deepcopy(self.dqn)

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)

        # Replay Memory
        self.replay = ReplayMemory()

        # Epsilon
        self.epsilon = EPSILON_START

    def select_action(self, states: np.array) -> tuple:
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                       math.exp(-1. * self.step / EPSILON_DECAY)

        if self.epsilon > random():
            # Random Action
            available_actions = states[0, 1] == 0
            sample_action = np.random.randint(0, np.count_nonzero(available_actions))
            sample_action = available_actions.nonzero()[0][sample_action]
            action = torch.LongTensor([[sample_action]])
            return action
        actions = states[0][1].reshape(1, -1)  # assumes self.action_repeat = 1
        states = states[0][0].reshape(1, self.action_repeat, self.env.width, self.env.height)
        states_variable: (Variable, Variable) = (
            Variable(torch.FloatTensor(states).cuda()), Variable(torch.FloatTensor(actions).cuda()))

        if self.mode == 'dqn':
            # states_variable.volatile = True
            action = self.dqn(states_variable).data.cpu().max(1)[1].unsqueeze(0)
        elif self.mode == 'lstm':
            # make sure the hidden states are ok
            action, self.dqn_hidden_state, self.dqn_cell_state = \
                self.dqn(states_variable, self.train_hidden_state, self.train_cell_state)
            action = action.data.cpu().max(1)[1].unsqueeze(0)

        return action

    def get_initial_states(self):
        state = self.env.reset()
        state = self.env.get_screen()
        states = np.stack([state], axis=0)

        self._state_buffer = deque(maxlen=self.action_repeat)
        self._state_buffer.append(state)
        return states

    def add_state(self, state):
        self._state_buffer.append(state)

    def recent_states(self):
        return np.array(self._state_buffer)

    def train(self, gamma: float = 0.99):
        # Initial States
        reward_sum = 0.
        q_mean = [0., 0.]
        target_mean = [0., 0.]

        for i in range(EPOCHS * TRAIN_SIZE):
            if i % TRAIN_SIZE == 0:
                print('starting epoch', i // TRAIN_SIZE)

            # Init LSTM States
            if self.mode == 'lstm':
                # For Training
                self.train_hidden_state, self.train_cell_state = self.dqn.reset_states(self.train_hidden_state,
                                                                                       self.train_cell_state)

            states = self.get_initial_states()
            losses = []
            checkpoint_flag = False
            target_update_flag = False
            play_steps = 0
            real_play_count = 0
            real_score = 0

            reward = 0
            reward_sum = 0
            done = False
            while True:
                # Get Action
                action: torch.LongTensor = self.select_action(states)

                observation, reward, done = self.env.step(action[0, 0])
                next_state = self.env.get_screen()
                self.add_state(next_state)

                reward_sum += reward

                # Store the infomation in Replay Memory
                next_states = self.recent_states()
                if done:
                    self.replay.put(states, action, reward, None)
                else:
                    self.replay.put(states, action, reward, next_states)

                # Change States
                states = next_states

                # Optimize
                if self.replay.is_available():
                    # loss, reward_sum, q_mean, target_mean = self.optimize(gamma)
                    loss, _, _, _ = self.optimize(gamma)
                    losses.append(loss)

                if done:
                    break

                # Increase step
                self.step += 1
                play_steps += 1

                if self.step % 100 == 0:
                    print('step:', self.step)

                # Target Update
                if self.step % TARGET_UPDATE_INTERVAL == 0:
                    self._target_update()
                    target_update_flag = True

                # Checkpoint - uncomment if want to save each CHECKPOINT_INTERVAL (o/w saves when model improves)
                # if self.step % CHECKPOINT_INTERVAL == 0:
                #     self.save_checkpoint(filename=f'dqn_checkpoints/chkpoint_{self.mode}_{self.step}.pth.tar')
                #     checkpoint_flag = True

                # Play
                if i + 1 % PLAY_INTERVAL == 0:

                    self.env.set_mode('eval')

                    scores = []
                    counts = []
                    for j in range(PLAY_REPEAT):
                        score, real_play_count = self.play(logging=False)
                        print("play example %d finished with score %.3f" % (j, score))
                        scores.append(score)
                        counts.append(real_play_count)
                        logger.debug(f'[{self.step}] [Validation] play_score: {score}, play_count: {real_play_count}')
                    real_score = int(np.mean(scores))
                    real_play_count = int(np.mean(counts))

                    if self.best_score <= real_score:
                        self.best_score = real_score
                        self.best_count = real_play_count
                        logger.debug(f'[{self.step}] [CheckPoint] Play: {self.best_score} [Best Play] [checkpoint]')
                        self.save_checkpoint(
                            filename=f'dqn_checkpoints/chkpoint_{self.mode}_{self.best_score}.pth.tar')

                    logger.info(
                        f'[{self.step}] [Validation] mean_score: {real_score}, mean_play_count: {real_play_count}')

                    self.env.set_mode('train')

            print('finished example %d in %d steps with total reward of %d' % (i + 1, play_steps, reward_sum))

            # Logging
            mean_loss = np.mean(losses)
            target_update_msg = '  [target updated]' if target_update_flag else ''
            # save_msg = '  [checkpoint!]' if checkpoint_flag else ''
            logger.info(f'[{self.step}] Loss:{mean_loss:<8.4} Play:{play_steps:<3}  '
                        f'Epsilon:{self.epsilon:<6.4}{target_update_msg}    '
                        f'RewardSum:{reward_sum:<3}')  # Q:[{q_mean[0]:<6.4}, {q_mean[1]:<6.4}] '
            # f'T:[{target_mean[0]:<6.4}, {target_mean[1]:<6.4}] '

    def optimize(self, gamma: float):
        if self.mode == 'lstm':
            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.dqn.reset_states(self.dqn_hidden_state,
                                                                               self.dqn_cell_state)
            self.target_hidden_state, self.target_cell_state = self.dqn.reset_states(self.target_hidden_state,
                                                                                     self.target_cell_state)

        # Get Sample
        transitions = self.replay.sample(BATCH_SIZE)

        # Mask
        non_final_mask = torch.tensor(list(map(lambda ns: ns is not None, transitions.next_state)), dtype=bool).cuda()
        final_mask = ~non_final_mask

        state_batch: Variable = Variable(torch.cat([ns[0] for ns in transitions.state]).cuda())
        state_actions_batch: Variable = Variable(torch.cat([ns[1] for ns in transitions.state]).cuda())
        action_batch: Variable = Variable(torch.cat(transitions.action).cuda())
        reward_batch: Variable = Variable(torch.cat(transitions.reward).cuda())
        non_final_next_state_batch = Variable(
            torch.cat([ns[0] for ns in transitions.next_state if ns is not None]).cuda())
        non_final_next_state_action_batch = Variable(
            torch.cat([ns[1] for ns in transitions.next_state if ns is not None]).cuda())
        # non_final_next_state_batch.volatile = True

        # Reshape States and Next States
        state_batch = state_batch.view([BATCH_SIZE, self.action_repeat, self.env.width, self.env.height])
        state_actions_batch = state_actions_batch.view(
            [BATCH_SIZE, self.env.action_space])  # assumes self.action_repeat = 1
        state_batch = (state_batch, state_actions_batch)
        non_final_next_state_batch = non_final_next_state_batch.view(
            [-1, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_action_batch = non_final_next_state_action_batch.view(
            [-1, self.env.action_space])  # assumes self.action_repeat = 1
        non_final_next_state_batch = (non_final_next_state_batch, non_final_next_state_action_batch)
        # non_final_next_state_batch.volatile = True

        # Clipping Reward between -1 and 1
        # reward_batch.data.clamp_(-1, 1) #consider adding some clamping

        # Predict by DQN Model
        if self.mode == 'dqn':
            q_pred = self.dqn(state_batch)
        elif self.mode == 'lstm':
            q_pred, self.dqn_hidden_state, self.dqn_cell_state = self.dqn(state_batch, self.dqn_hidden_state,
                                                                          self.dqn_cell_state)

        q_values = q_pred.gather(1, action_batch)

        # Predict by Target Model
        target_values = Variable(torch.zeros(BATCH_SIZE, 1).cuda())
        if self.mode == 'dqn':
            target_pred = self.target(non_final_next_state_batch)
        elif self.mode == 'lstm':
            target_pred, self.target_hidden_state, self.target_cell_state = self.target(non_final_next_state_batch,
                                                                                        self.target_hidden_state,
                                                                                        self.target_cell_state)

        target_values[non_final_mask] = reward_batch.unsqueeze(1)[non_final_mask] + (
                target_pred.max(1)[0] * gamma).unsqueeze(1)
        target_values[final_mask] = reward_batch[final_mask].unsqueeze(1).detach()

        loss = F.smooth_l1_loss(q_values, target_values)  # consider l2 loss (mean or sum)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if self.clip:
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        reward_score = int(torch.sum(reward_batch).data.cpu().numpy())
        q_mean = torch.sum(q_pred, 0).data.cpu().numpy()
        target_mean = torch.sum(target_pred, 0).data.cpu().numpy()

        return loss.data.cpu().numpy(), reward_score, q_mean, target_mean

    def _target_update(self):
        self.target = copy.deepcopy(self.dqn)

    def save_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar'):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        checkpoint = {
            'dqn': self.dqn.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'best': self.best_score,
            'best_count': self.best_count
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename='/trained_models/dqn_checkpoints/checkpoint.pth.tar', epsilon=None):
        checkpoint = torch.load(filename)
        self.dqn.load_state_dict(checkpoint['dqn'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.best_score = self.best_score or checkpoint['best']
        self.best_count = checkpoint['best_count']

    def load_latest_checkpoint(self, epsilon=None):
        r = re.compile('chkpoint_(dqn|lstm)_(?P<number>-?\d+)\.pth\.tar$')

        files = glob.glob(f'dqn_checkpoints/chkpoint_{self.mode}_*.pth.tar')

        if files:
            files = list(map(lambda x: [int(r.search(x).group('number')), x], files))
            files = sorted(files, key=lambda x: x[0])
            latest_file = files[-1][1]
            self.load_checkpoint(latest_file, epsilon=epsilon)
            print(f'latest checkpoint has been loaded - {latest_file}')
        else:
            print('no latest checkpoint')

    def play(self, logging=True):
        states = self.get_initial_states()
        count = 0
        total_score = 0

        if self.mode == 'lstm':
            self.test_hidden_state, self.test_cell_state = self.dqn.reset_states(self.test_hidden_state,
                                                                                 self.test_cell_state)

        while True:

            actions = states[0][1].reshape(1, -1)  # assumes self.action_repeat = 1
            states = states[0][0].reshape(1, self.action_repeat, self.env.width, self.env.height)
            states_variable: (Variable, Variable) = (
                Variable(torch.FloatTensor(states).cuda()), Variable(torch.FloatTensor(actions).cuda()))

            if self.mode == 'dqn':
                dqn_pred = self.dqn(states_variable)
            elif self.mode == 'lstm':
                dqn_pred, self.test_hidden_state, self.test_cell_state = \
                    self.dqn(states_variable, self.test_hidden_state, self.test_cell_state)

            action = dqn_pred.data.cpu().max(1)[1].unsqueeze(0)

            for _ in range(self.frame_skipping):
                observation, reward, done = self.env.step(action)
                # States <- Next States
                next_state = self.env.get_screen()
                self.add_state(next_state)
                states = self.recent_states()

                total_score += reward

                if done:
                    break

            # Logging
            count += 1
            if logging:
                action_dist = torch.sum(dqn_pred, 0).data.cpu().numpy()[0]
                print(f'[{count}] action:{action} {action_dist}, reward:{reward}')

            if done:
                break
        return total_score, count

    def inspect(self):
        print(dir(self.dqn.conv1))

        for param in list(self.dqn.parameters()):
            print(param.size())

    def _sum_params(self, model):
        return np.sum([torch.sum(p).data[0] for p in model.parameters()])


def main(parser):
    agent = Agent(parser, cuda=use_cuda)
    if parser.load_latest and not parser.checkpoint:
        agent.load_latest_checkpoint()
    elif parser.checkpoint:
        agent.load_checkpoint(parser.checkpoint)

    if parser.mode.lower() == 'play':
        agent.play()
    elif parser.mode.lower() == 'train':
        agent.train()
    elif parser.mode.lower() == 'inspect':
        agent.inspect()


if __name__ == '__main__':
    main(parser)
