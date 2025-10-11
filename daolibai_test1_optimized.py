import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

# 参数
epsilon = 0.5
alpha = 0.1
gamma = 0.9

# 奖励函数
rewards = np.array([-abs(i - 50) for i in range(101)])

# Q表初始化
q_table = np.load("q_table.npy") if os.path.exists("q_table.npy") else np.zeros((101, 15))

# 倒立摆参数
M = 1.0
m = 0.1
l = 1.0
g = 9.81

# 动力学方程
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

# 训练50次
for run in range(100):
    state = np.array([0.0, 0.0, 5 * np.pi / 180, 0.0])
    datatrans = []

    for step in range(200):
        current_state = np.clip(int(state[2] * 180 / np.pi), 0, 100)
        
        if np.random.random() < epsilon:
            current_action = np.random.randint(0, 15)
        else:
            current_action = np.argmax(q_table[current_state])

        F = current_action - 7
        
        sol = solve_ivp(lambda t, y: dynamics(t, y, F), [0, 0.02], state, t_eval=[0.02])
        state = sol.y[:, -1]
        
        datatrans.append([state[2], state[3]])
        
        next_state = np.clip(int(state[2] * 180 / np.pi), 0, 100)
        max_next_q = np.max(q_table[next_state])
        q_table[current_state, current_action] += alpha * (
            rewards[next_state] + gamma * max_next_q - q_table[current_state, current_action]
        )

    # 每10次保存数据和Q表
    if (run + 1) % 10 == 0:
        np.savetxt(f"pendulum_data_run_{run+1}.csv", datatrans, delimiter=",", 
                   header="Angle (rad),Angular Velocity (rad/s)", comments="")
        np.save("q_table.npy", q_table)
        print(f"Run {run+1} completed")