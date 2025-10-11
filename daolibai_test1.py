import numpy as np
import pandas as pd
import random
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

# ---- 代码A: 深度学习控制部分（Q学习） ----

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

# 自动加载已有的 Q 表
def load_q_table():
    if os.path.exists("q_table.pkl"):
        return pd.read_pickle("q_table.pkl")
    else:
        return pd.DataFrame(data=[[0 for _ in actions] for _ in states], index=states, columns=actions)

q_table = load_q_table()

def get_next_state(state, action):
    if action < 10 and 0 <= state + action <= 100:
        return state + action
    return state

def get_valid_actions(state):
    if degree > 0:
        return [a for a in actions[10:] if a < len(actions)]
    else:
        return [a for a in actions[:10] if 0 <= state + a <= 100]

# ---- 代码B: 倒立摆动力学模拟部分 ----

# 参数
M = 1.0   # 车质量
m = 0.1   # 杆质量
l = 1.0   # 摆长
g = 9.81  # 重力加速度

# 状态变量 [x v theta omega]
state = np.array([0.0, 0.0, 5 * np.pi / 180, 0.0])  # 5度初始角度

# 控制力（由Q学习控制）
F = 0

datatrans = []

# 动力学方程
def dynamics(t, s, F):
    x, v, th, w = s
    b = 0.05  # 小车阻力
    I = (1/3) * m * (2 * l)**2  # 杆的转动惯量
    
    def equations(vars):
        xddot, wdot = vars
        x_gc_ddot = xddot + l * wdot * np.cos(th) - l * w**2 * np.sin(th)
        y_gc_ddot = l * wdot * np.sin(th) + l * w**2 * np.cos(th)
        N = m * x_gc_ddot
        P = m * g + m * y_gc_ddot
        eq1 = F - M * xddot - N - b * v
        eq2 = N * np.cos(th) - P * np.sin(th) - I * wdot / (2 * l)
        return [eq1, eq2]

    xddot, wdot = fsolve(equations, [0, 0], xtol=1e-8, maxfev=5000)
    return [v, xddot, w, wdot]

# 训练50次
for run in range(50):
    state = np.array([0.0, 0.0, 5 * np.pi / 180, 0.0])  # 每次训练前重置初始状态
    datatrans.clear()  # 清空数据记录

    for step in range(200):  # 每次训练200个步数
        # 当前状态，假设我们通过状态变量获取某种量作为“状态”输入
        current_state = int(state[2] * 180 / np.pi)  # 假设用摆角度作为离散状态（0-100）
        current_state = max(0, min(100, current_state))

        # 基于Q学习策略选择动作
        if random.uniform(0, 1) < epsilon:
            valid_actions = get_valid_actions(current_state)
            if valid_actions:
                current_action = random.choice(valid_actions)
            else:
                current_action = 0  # 默认动作
        else:
            current_action = q_table.loc[current_state].idxmax()

        # 根据动作获取控制力F
        F = current_action - 7  # 将动作映射到控制力

        # 使用倒立摆模型更新状态
        sol = solve_ivp(lambda t, y: dynamics(t, y, F), [0, 0.02], state, t_eval=[0.02])
        state = sol.y[:, -1]

        # 保存角度和角速度
        theta = state[2]
        omega = state[3]
        datatrans.append([theta, omega])

        # 更新 Q 表
        next_state = int(state[2] * 180 / np.pi)
        next_state = max(0, min(100, next_state))
        valid_next_actions = get_valid_actions(next_state)
        max_next_q = q_table.loc[next_state, valid_next_actions].max() if valid_next_actions else 0
        q_table.loc[current_state, current_action] = int(q_table.loc[current_state, current_action] + alpha * (
            rewards[next_state] + gamma * max_next_q - q_table.loc[current_state, current_action]
        ))

    # 保存数据
    df = pd.DataFrame(datatrans, columns=["Angle (rad)", "Angular Velocity (rad/s)"])
    df.to_csv(f"pendulum_data_run_{run+1}.csv", index=False)

    # 每次训练结束后，保存 Q 表
    q_table.to_pickle("q_table.pkl")
