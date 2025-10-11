# 引入函数库

import pandas as pd
import numpy as np
import random
import time

# 参数
epsilon = 0.5   # 以 epsilon 的概率进行探索
alpha = 0.1     # 学习率
gamma = 0.9     # 奖励递减值

# 状态和动作空间
states = list(range(101))  # 状态集，从 0 到 100
actions = list(range(15))  # 动作集 0-14
degree = 0
degree_v = 0

# 奖励函数
rewards = {state: -abs(state - 50) for state in states}  # 目标是状态50

# Q表初始化
q_table = pd.DataFrame(data=[[0 for _ in actions] for _ in states], index=states, columns=actions)

def get_next_state(state, action):
    if action < 10 and 0 <= state + action <= 100:
        return state + action
    return state

def get_valid_actions(state):
    if degree > 0:
        return [a for a in actions[10:] if a < len(actions)]
    else:
        return [a for a in actions[:10] if 0 <= state + a <= 100]

history = np.zeros((1000, 50))

for i in range(50):
    sumreward = 0
    current_state = 70
    total_steps = 0
    
    while sumreward > -100 and total_steps < 999:  # 防止无限循环
        if (random.uniform(0, 1) < epsilon) or ((q_table.loc[current_state] == 0).all()):
            valid_actions = get_valid_actions(current_state)
            if valid_actions:
                current_action = random.choice(valid_actions)
            else:
                break
        else:
            current_action = q_table.loc[current_state].idxmax()

        next_state = get_next_state(current_state, current_action)
        valid_next_actions = get_valid_actions(next_state)
        
        if valid_next_actions:
            next_state_q_values = q_table.loc[next_state, valid_next_actions]
            max_next_q = next_state_q_values.max()
        else:
            max_next_q = 0

        q_table.loc[current_state, current_action] += alpha * (
            rewards[next_state] + gamma * max_next_q - q_table.loc[current_state, current_action])
        
        current_state = next_state
        history[total_steps, i] = current_state
        total_steps += 1
        sumreward += rewards[next_state]

    print(f'Episode {i}: steps={total_steps}, reward={sumreward:.2f}')

print('\n q_table:')
print(q_table)