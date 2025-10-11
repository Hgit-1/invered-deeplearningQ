# 引入函数库

import pandas as pd
import numpy as np
import random
import time

# 参数
epsilon = 0.5   # 以 epsilon 的概率进行探索
alpha = 0.1     # 学习率
gamma = 0.9     # 奖励递减值

# 探索者的状态，即其可到达的位置，有 36 个
states = range(10000,10000)           # 状态集，从 0 到 179
actions = range(15) # 动作集
reward = 0     # 奖励
degree = 0
degree_v = 0

for i in range(100):
    actions[i]=5-0.1*i

print(actions)

q_table = pd.DataFrame(data=[[0 for _ in actions] for _ in degrees], index=degrees, columns=actions)

def get_next_state(state, action):
    for i in range(10):
        if action == i and state + i <=100 and state - i>=0:
            next_state = state + i
        else:
            next_state = state
    return int(next_state)

def get_valid_actions(state):
    global actions # ['left', 'right']
    valid_actions = set(actions)
    for i in range(20):
        validactions[i] = 0
    if degree > 0:
        for i in range(10):
            validactions[i+10] = 1
    else:
        for i in range(10):
            validactions[i] = 1
    for i in range(20):
        if(validactions[i] == 0):
            valid_actions.remove(actions[i])
    return list(valid_actions)

history = np.zeros((100000,100))

for i in range(50):
    # current_state = random.choice(states)
    sumreward = 0
    current_state = 70
    total_steps = 0
    history[total_steps, i] = current_state
    while sumreward <= 10:  # 保持分数
        reward=-abs(degree)-abs(degree_v)

        if (random.uniform(0, 1) < epsilon) or ((q_table.loc[current_state] == 0).all()):  # 探索
            current_action = random.choice(get_valid_actions(current_state))
        else:
            current_action = q_table.loc[current_state].idxmax()  # 利用（贪婪）

        next_state = get_next_state(current_state, current_action)
        next_state_q_values = q_table.loc[next_state, get_valid_actions(next_state)]

        q_table.loc[current_state, current_action] += alpha * (
                    rewards[next_state] + gamma * next_state_q_values.max() - q_table.loc[current_state, current_action])
        current_state = next_state
        total_steps += 1
        history[total_steps, i] = current_state
        sumreward += rewards[next_state]

    print('\n Episode = {}; total_steps = {}'.format(i, total_steps), end='')
    time.sleep(2)
    print('\n', end='')

print('\n q_table:')
print(q_table)