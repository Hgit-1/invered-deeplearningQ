import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

# 参数
epsilon = 1.0
alpha = 0.02
gamma = 0.9

# 奖励函数
rewards = np.array([-abs(i - 180) for i in range(361)]) #！

# Q表初始化
q_table = np.load("q_table.npy") if os.path.exists("q_table.npy") else np.zeros((361, 15))

# 倒立摆参数
M = 1.0
m = 0.1
l = 1.0
g = 9.81

# x=位置，v=速度，th=角度，w=角速度
def dynamics(t, s, F):
    x, v, th, w = s
    b = 0.05
    I = (1/3) * m * (2 * l)**2
    
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
    state = np.array([0.0, 0.0, 3, 0.0]) 
    datatrans = []

    step=0
    total_reward = 0
    while True: #！
        # 角度状态离散化：将[-180°,+180°]映射到[0,100]，0°对应索引50 #！
        step += 1
        #angle_deg = state[2] * 180 / np.pi
        #current_state = np.clip(int(angle_deg * 100/360 + 50), 0, 100)
        current_state = int(state[2])

        # ε-greedy 策略
        if np.random.random() < epsilon:
            current_action = np.random.randint(0, 15)
        else:
            current_action = np.argmax(q_table[current_state])

        # 基于系统参数的动作映射：力范围基于重力和系统质量
        F_max = 2 * (M + m) * g / M  
        F = (current_action - 7) * F_max / 7  # 映射

        # 状态更新
        sol = solve_ivp(lambda t, y: dynamics(t, y, F), [0, 0.02], state, t_eval=[0.02])
        state = sol.y[:, -1]
        
        # 状态边界
        state[0] = np.clip(state[0], -1000, 1000)  # 限制位置
        state[1] = np.clip(state[1], -10, 10)  # 限制速度
        state[2] = np.clip(state[2], -30, 30)  # 限制角度
        state[3] = np.clip(state[3], -20, 20)  # 限制角速度

        datatrans.append([state[2], state[3]])

        # 奖励函数
        #angle_deg = state[2] * 180 / np.pi
        next_state = np.clip(int(state[2]), -30, 30)
        
        # 角度的简单奖励
        reward = -abs(state[2])-abs(state[3])
        total_reward += reward 
        
        # 位置约束惩罚
        if abs(state[0]) == 30:
            total_reward -= 10
            break

        # 速度约束惩罚
        if abs(state[1]) == 10:
            total_reward -= 10

        if total_reward < -1000:
            break
        # Q 学习更新
        max_next_q = np.max(q_table[next_state])
        q_table[current_state, current_action] += alpha * (
            total_reward + gamma * max_next_q - q_table[current_state, current_action]
        )

    # 保存
    if (run + 1) % 10 == 0:
        np.savetxt(f"pendulum_data_run_{run+1}.csv", datatrans, delimiter=",", 
                   header="Angle (rad),Angular Velocity (rad/s)", comments="")
        np.save("q_table.npy", q_table)
        print(f"Run {run+1} completed")
