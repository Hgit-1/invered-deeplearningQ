import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

# 参数
epsilon = 1.0  # 探索率
epsilon_decay = 0.995  # epsilon 衰减率
alpha = 0.02  # 学习率
gamma = 0.9  # 折扣因子
max_steps = 200  # 每回合最大步数

# 奖励函数 - 使用角度和角速度的简单惩罚
def reward_function(state):
    # 简单的奖励函数：惩罚角度和角速度
    angle_penalty = -abs(state[2])  # 角度惩罚
    velocity_penalty = -abs(state[3])  # 角速度惩罚
    return angle_penalty + velocity_penalty

# Q表初始化
q_table = np.load("q_table.npy") if os.path.exists("q_table.npy") else np.zeros((101, 15))  # 角度离散为 101 个状态

# 倒立摆参数
M = 1.0  # 车质量
m = 0.1  # 摆杆质量
l = 1.0  # 摆杆长度
g = 9.81  # 重力加速度

# x=位置，v=速度，th=角度，w=角速度
def dynamics(t, s, F):
    x, v, th, w = s
    b = 0.05  # 摩擦系数
    I = (1/3) * m * (2 * l)**2  # 摆杆转动惯量
    
    def equations(vars):
        xddot, wdot = vars
        x_gc_ddot = xddot + l * wdot * np.cos(th) - l * w**2 * np.sin(th)
        y_gc_ddot = l * wdot * np.sin(th) + l * w**2 * np.cos(th)
        N = m * x_gc_ddot
        P = m * g + m * y_gc_ddot
        eq1 = F - M * xddot - N - b * v
        eq2 = N * np.cos(th) - P * np.sin(th) - I * wdot / (2 * l)
        return [eq1, eq2]

    xddot, wdot = fsolve(equations, [0, 0], xtol=1e-6, maxfev=10000)
    return [v, xddot, w, wdot]

# 训练 100 回合
for run in range(100):
    state = np.array([0.0, 0.0, 3 * np.pi / 180, 0.0])  # 初始状态，角度为3度
    datatrans = []

    for step in range(max_steps):
        # 角度状态离散化：将[-180°,+180°]映射到[0,100]，0°对应索引50
        angle_deg = state[2] * 180 / np.pi  # 将角度转换为度数
        current_state = np.clip(int(angle_deg * 100 / 360 + 50), 0, 100)  # 映射到[0, 100]的状态

        # ε-greedy 策略
        if np.random.random() < epsilon:
            current_action = np.random.randint(0, 15)  # 随机选择一个动作
        else:
            current_action = np.argmax(q_table[current_state])  # 选择Q值最大的动作

        # 基于系统参数的动作映射：力范围基于重力和系统质量
        F_max = 2 * (M + m) * g / M  
        F = (current_action - 7) * F_max / 7  # 将动作映射到力的范围

        # 状态更新
        sol = solve_ivp(lambda t, y: dynamics(t, y, F), [0, 0.02], state, t_eval=[0.02])
        state = sol.y[:, -1]

        # 状态边界
        state[0] = np.clip(state[0], -1000, 1000)  # 限制位置
        state[1] = np.clip(state[1], -10, 10)  # 限制速度
        state[2] = np.clip(state[2], -np.pi, np.pi)  # 限制角度
        state[3] = np.clip(state[3], -20, 20)  # 限制角速度

        datatrans.append([state[2], state[3]])

        # 奖励函数
        reward = reward_function(state)
        total_reward = reward
        
        # Q 学习更新
        angle_deg = state[2] * 180 / np.pi  # 转换为度数
        next_state = np.clip(int(angle_deg * 100 / 360 + 50), 0, 100)

        max_next_q = np.max(q_table[next_state])
        q_table[current_state, current_action] += alpha * (
            total_reward + gamma * max_next_q - q_table[current_state, current_action]
        )

        # epsilon 衰减
        epsilon *= epsilon_decay

        # 位置约束惩罚
        if abs(state[0]) >= 30:
            total_reward -= 10  # 如果位置超过限制，惩罚

        # 如果奖励过低，终止训练
        if total_reward < -1000:
            break

    # 保存
    if (run + 1) % 10 == 0:
        np.savetxt(f"pendulum_data_run_{run+1}.csv", datatrans, delimiter=",", 
                   header="Angle (rad),Angular Velocity (rad/s)", comments="")
        np.save("q_table.npy", q_table)
        print(f"Run {run+1} completed")