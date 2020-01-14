import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from collections import namedtuple
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

Transition = namedtuple('Transition', 'state, action, next_state, reward')

class MLP(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        # self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

class DQNAgent(object):
    def __init__(self, n_state, n_action, args):
        self.online = MLP(n_state, args.num_hidden, n_action)
        self.target = MLP(n_state, args.num_hidden, n_action)
        self.args = args
        self.replay_buffer = ReplayBuffer(args.max_size)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=args.lr)

        make_update_exp(self.online, self.target, rate=1.0)

    def act(self, state):
        output = self.online(torch.tensor(state[None], dtype=torch.float32))
        epsilon = 1e-6
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(output, dim=1).numpy()[0]
        return action
        

    def update(self, transition):
        self.replay_buffer.add(transition)
        if len(self.replay_buffer) < self.args.batch_size:
            return None, None, None, None
        
        transitions = self.replay_buffer.sample(self.args.batch_size)
        state = torch.tensor(transitions.state, dtype=torch.float32)
        action = torch.tensor(transitions.action, dtype=torch.long).unsqueeze(1)
        next_state = torch.tensor(transitions.next_state, dtype=torch.float32)
        reward = torch.tensor(transitions.reward, dtype=torch.float32)
        
        q = torch.gather(self.online(state), dim=1, index=action)
        next_q = torch.max(self.target(next_state), dim=1)[0].detach()
        debug_q = [torch.mean(torch.gather(self.online(state), dim=1, index=action)).item(), 
                    torch.mean(torch.gather(self.online(state), dim=1, index=(1-action))).item()]
        # debug_q = [torch.mean(q).item(), 
        #             torch.mean(torch.gather(self.online(state), dim=1, index=(1-action))).item()]
        
        td_error = reward + self.args.gamma * next_q - q


        loss = torch.mean(torch.pow(td_error, 2))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        make_update_exp(self.online, self.target, rate=self.args.rate)

        

        return (loss.item(), torch.mean(q).item(), torch.mean(next_q).item(), debug_q)


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = int(max_size)
        self.index = 0
    
    def add(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer[(self.index + 1) % self.max_size] = transition
        else:
            self.buffer.append(transition)
        self.index = (self.index + 1) % self.max_size

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, int(batch_size))
        transitions = Transition(*(zip(*samples)))
        
        return transitions


class Args(object):
    def __init__(self):
        self.num_hidden = 10
        self.batch_size = 100
        self.max_size = 1e6
        self.lr = 1e-5
        self.gamma = 0.9
        self.rate = 1e-3


def make_update_exp(source, target, rate=1e-2):
    """ Use values of parameters from the source model to update values of parameters from the target model. Each update just change values of paramters from the target model slightly, which aims to provide relative stable evaluation. Note that structures of the two models should be the same. 
    
    Parameters
    ----------
    source : torch.nn.Module
        The model which provides updated values of parameters.
    target : torch.nn.Module
        The model which receives updated values of paramters to update itself.
    """
    polyak = rate
    for tgt, src in zip(target.named_parameters(recurse=True), source.named_parameters(recurse=True)):
        assert src[0] == tgt[0] # The identifiers should be the same
        tgt[1].data = polyak * src[1].data + (1.0 - polyak) * tgt[1].data

def display(agent, env):
    display_epoch = 2
    for i in range(display_epoch):
        print('i:{}'.format(i))
        obs = env.reset()
        while True:
            action = agent.act(obs)
            _, _ , done, info = env.step(action)
            env.render()
            time.sleep(0.2)
            if done:
                obs = env.reset()
                break

def train():
    env = gym.make('CartPole-v0')
    env.seed(1)
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    agent = DQNAgent(n_state, n_action, Args())
    
    
    episode_reward = [0.0]
    epoch = 0
    losses = [0.0]
    q_values = [0.0]
    next_q_values = [0.0]
    difference_q_next_q = [0.0]
    debug_qs = [[0.0, 0.0]]
    difference_q = [0.0]
    
    obs = env.reset()
    gap = 100
    steps = 1
    first_update = True
    n_step = 1
    max_epoch = 2000
    start_time = time.time()
    total_time = start_time
    scale = 16.0

    max_result = 0
    while True:
        action = agent.act(obs * scale)
        next_obs, reward, done, _ = env.step(action)
        

        if done:
            reward_replay = reward
        else:
            reward_replay = reward

        transition = Transition(state=obs * scale, action=action, next_state=next_obs * scale, reward=reward_replay)

        if done:
            obs = env.reset()
            epoch += 1
            n_step = 1
            episode_reward[-1] += reward
            if epoch % 500 == 0:
                print('Time elapses:{:.1f}; Total Time:{:.1f}min'.format(time.time()-start_time, (time.time()-total_time) / 60))
                start_time = time.time()   
            if np.mean(episode_reward[-gap:]) > max_result:
                max_result = np.mean(episode_reward[-gap:])       
            if epoch >= max_epoch or np.mean(episode_reward[-gap:]) >= 300:
                # display(agent, env)
                print('max result:{}'.format(max_result))
                break
            else:
                episode_reward.append(0.0)
                losses.append(0.0)
                q_values.append(0.0)
                next_q_values.append(0.0)
                difference_q_next_q.append(0.0)
                debug_qs.append([0.0, 0.0])
                difference_q.append(0.0)
        else:
            obs = next_obs
            episode_reward[-1] += reward
        loss, q_value, next_q_value, debug_q = agent.update(transition)
        if loss == None:
            losses[-1] =1.0
            q_values[-1] += 0.0
            next_q_values[-1] += 0.0
            debug_qs[-1][0] += 0.0
            debug_qs[-1][1] += 0.0
        else:
            if first_update:
                first_update = False
                print('First update: epoch {}; steps {}'.format(epoch + 1, steps + 1))
            losses[-1] = losses[-1] + (loss - losses[-1]) / n_step
            q_values[-1] = q_values[-1] + (q_value - q_values[-1]) / n_step
            next_q_values[-1] = next_q_values[-1] + (next_q_value - next_q_values[-1]) / n_step
            difference_q_next_q[-1] = difference_q_next_q[-1] + (np.abs(q_value - next_q_value) - difference_q_next_q[-1]) / n_step
            debug_qs[-1][0] = debug_qs[-1][0] + (debug_q[0] - debug_qs[-1][0]) / n_step
            debug_qs[-1][1] = debug_qs[-1][1] + (debug_q[1] - debug_qs[-1][1]) / n_step
            difference_q[-1] = difference_q[-1] + (np.abs(debug_q[0] - debug_q[1]) - difference_q[-1]) / n_step
            # losses[-1] += loss           
            # q_values[-1] += q_value
        steps += 1 
        n_step += 1  

    np.savetxt('episode_reward.txt', episode_reward, fmt='%.8lf', encoding='utf-8')
    np.savetxt('losses.txt', losses, fmt='%.8lf', encoding='utf-8')
    np.savetxt('q_values.txt', q_values, fmt='%.8lf', encoding='utf-8')
    np.savetxt('next_q_values.txt', next_q_values, fmt='%.8lf', encoding='utf-8')
    np.savetxt('debug_qs.txt', debug_qs, fmt='%.8lf', encoding='utf-8')
    np.savetxt('difference.txt', difference_q_next_q, fmt='%.8lf', encoding='utf-8')
    
    episode_reward = [np.mean(episode_reward[max(0,s-gap):s]) for s in range(0, len(episode_reward))]
    # losses = [np.mean(losses[max(0,s-gap):s]) for s in range(0, len(losses))]
    # q_values = [np.mean(q_values[max(0,s-gap):s]) for s in range(0, len(q_values))]
    # next_q_values = [np.mean(next_q_values[max(0,s-gap):s]) for s in range(0, len(next_q_values))]
    # difference_q_next_q = [np.mean(difference_q_next_q[max(0,s-gap):s]) for s in range(0, len(difference_q_next_q))]
    # difference_q = [np.mean(difference_q[max(0,s-gap):s]) for s in range(0, len(difference_q))]

    debug_q_correct = []
    debug_q_wrong = []
    for c, w in debug_qs:
        debug_q_correct.append(c)
        debug_q_wrong.append(w)
    debug_qs = [[np.mean(debug_q_correct[s-gap:s]), np.mean(debug_q_wrong[s-gap:s])] for s in range(gap, len(debug_qs))]
    print('total steps:{}'.format(steps))
    plt.subplot(321)
    plt.plot(episode_reward)
    plt.title('Mean reward per 100 episode')
    plt.subplot(322)
    plt.plot(np.log10(np.sqrt(np.abs(losses))))
    plt.title('Training loss(log10)')
    plt.subplot(323)
    plt.plot(q_values, label='q-value')
    plt.plot(next_q_values, label='next-q-value')
    plt.legend()
    plt.title('Current action value and next action value')
    plt.subplot(324)
    plt.plot(np.log10(np.abs(difference_q_next_q)))
    plt.title('Difference between current action value and next action value (log 10)')
    plt.subplot(325)
    plt.plot([x[0] for x in debug_qs], label='q-value of correct action')
    plt.plot([x[1] for x in debug_qs], label='q-value of wrong action')
    plt.title('Action values of coorect action and wrong action')
    plt.legend()
    plt.subplot(326)
    plt.plot(np.log10(np.abs(difference_q)))
    plt.title('Difference between action values of correct action and wrong action (log10)')
    plt.show()

    plt.clf()
    plt.plot(episode_reward)
    plt.title('Mean reward per 100 episode', fontsize=16)
    plt.tick_params(labelsize=12)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Mean Reward', fontsize=16)
    plt.savefig('episode.pdf')
    plt.clf()
    plt.plot(np.log10(np.sqrt(np.abs(losses))))
    plt.title('Training loss(log10)', fontsize=16)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.tick_params(labelsize=12)
    plt.savefig('loss.pdf')
    plt.clf()
    plt.plot(q_values, label='q-value')
    plt.plot(next_q_values, label='next-q-value')
    plt.legend()
    plt.title('Current action value and next action value', fontsize=16)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Q-Value', fontsize=16)
    plt.tick_params(labelsize=12)
    plt.savefig('cn.pdf', fontsize=16)
    plt.clf()
    plt.plot(np.log10(np.abs(difference_q_next_q)))
    plt.title('Difference between current action value and next action value (log 10)', fontsize=12)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Difference', fontsize=16)
    plt.tick_params(labelsize=10)
    plt.savefig('dcn.pdf')
    plt.clf()
    plt.plot([x[0] for x in debug_qs], label='q-value of correct action')
    plt.plot([x[1] for x in debug_qs], label='q-value of wrong action')
    plt.title('Action values of coorect action and wrong action', fontsize=16)
    plt.legend()
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Q-Value', fontsize=16)
    plt.tick_params(labelsize=12)
    plt.savefig('cw.pdf')
    plt.clf()
    plt.plot(np.log10(np.abs(difference_q)))
    plt.title('Difference between action values of correct action and wrong action (log10)', fontsize=11)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Difference', fontsize=10)
    plt.tick_params(labelsize=10)
    plt.savefig('dcw.pdf')

    
if __name__ == "__main__":
    train()