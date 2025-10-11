import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os
from collections import deque

# ==================== 超参数配置 ====================
class Config:
    # Q学习参数
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    alpha = 0.1  # 提高学习率加速收敛
    alpha_decay = 0.9995
    alpha_min = 0.01
    gamma = 0.95  # 提高折扣因子重视长期奖励
    
    # 训练参数
    num_episodes = 200
    max_steps = 500
    
    # 状态空间离散化（增加维度）
    angle_bins = 51  # -180°到180°
    angular_vel_bins = 21  # -20到20 rad/s
    position_bins = 21  # -10到10 m
    velocity_bins = 11  # -5到5 m/s
    
    # 动作空间
    num_actions = 15
    
    # 物理参数
    M = 1.0
    m = 0.1
    l = 1.0
    g = 9.81
    b = 0.05
    dt = 0.02
    
    # 边界
    max_position = 10.0
    max_velocity = 5.0
    max_angle = np.pi
    max_angular_vel = 20.0

config = Config()

# ==================== 状态离散化 ====================
def discretize_state(state):
    """将连续状态映射到离散索引"""
    x, v, th, w = state
    
    # 角度：[-π, π] → [0, angle_bins-1]
    angle_idx = np.clip(
        int((th + np.pi) / (2 * np.pi) * config.angle_bins),
        0, config.angle_bins - 1
    )
    
    # 角速度：[-max_angular_vel, max_angular_vel] → [0, angular_vel_bins-1]
    w_idx = np.clip(
        int((w + config.max_angular_vel) / (2 * config.max_angular_vel) * config.angular_vel_bins),
        0, config.angular_vel_bins - 1
    )
    
    # 位置：[-max_position, max_position] → [0, position_bins-1]
    x_idx = np.clip(
        int((x + config.max_position) / (2 * config.max_position) * config.position_bins),
        0, config.position_bins - 1
    )
    
    # 速度：[-max_velocity, max_velocity] → [0, velocity_bins-1]
    v_idx = np.clip(
        int((v + config.max_velocity) / (2 * config.max_velocity) * config.velocity_bins),
        0, config.velocity_bins - 1
    )
    
    return (angle_idx, w_idx, x_idx, v_idx)

# ==================== 改进的奖励函数 ====================
def reward_function(state, action, next_state):
    """多目标奖励函数"""
    x, v, th, w = next_state
    
    # 1. 角度奖励（主要目标）
    angle_reward = -10 * th**2
    
    # 2. 角速度奖励（稳定性）
    angular_vel_reward = -w**2
    
    # 3. 位置惩罚（保持在中心）
    position_penalty = -0.1 * x**2
    
    # 4. 速度惩罚
    velocity_penalty = -0.1 * v**2
    
    # 5. 存活奖励
    survival_reward = 1.0
    
    # 6. 失败惩罚
    if abs(th) > np.pi / 6:  # 超过30度
        failure_penalty = -100
    elif abs(x) > config.max_position:
        failure_penalty = -100
    else:
        failure_penalty = 0
    
    total_reward = (angle_reward + angular_vel_reward + 
                   position_penalty + velocity_penalty + 
                   survival_reward + failure_penalty)
    
    return total_reward

# ==================== 动力学模型 ====================
def dynamics(t, s, F):
    """倒立摆动力学方程"""
    x, v, th, w = s
    I = (1/3) * config.m * (2 * config.l)**2
    
    def equations(vars):
        xddot, wdot = vars
        x_gc_ddot = xddot + config.l * wdot * np.cos(th) - config.l * w**2 * np.sin(th)
        y_gc_ddot = config.l * wdot * np.sin(th) + config.l * w**2 * np.cos(th)
        N = config.m * x_gc_ddot
        P = config.m * config.g + config.m * y_gc_ddot
        eq1 = F - config.M * xddot - N - config.b * v
        eq2 = N * np.cos(th) - P * np.sin(th) - I * wdot / (2 * config.l)
        return [eq1, eq2]
    
    xddot, wdot = fsolve(equations, [0, 0], xtol=1e-6, maxfev=10000)
    return [v, xddot, w, wdot]

# ==================== Q表初始化 ====================
q_table_shape = (config.angle_bins, config.angular_vel_bins, 
                 config.position_bins, config.velocity_bins, 
                 config.num_actions)

if os.path.exists("q_table_v4.npy"):
    q_table = np.load("q_table_v4.npy")
    print(f"加载已有Q表，形状: {q_table.shape}")
else:
    q_table = np.zeros(q_table_shape)
    print(f"创建新Q表，形状: {q_table.shape}")

# ==================== 训练循环 ====================
epsilon = config.epsilon
alpha = config.alpha
episode_rewards = []
episode_steps = []

print("开始训练...")
print(f"状态空间大小: {np.prod(q_table_shape[:-1]):,}")
print(f"动作空间大小: {config.num_actions}")
print("-" * 60)

for episode in range(config.num_episodes):
    # 初始化状态
    state = np.array([0.0, 0.0, np.random.uniform(-0.1, 0.1), 0.0])
    total_reward = 0
    datatrans = []
    
    for step in range(config.max_steps):
        # 获取离散状态
        discrete_state = discretize_state(state)
        
        # ε-greedy策略
        if np.random.random() < epsilon:
            action = np.random.randint(0, config.num_actions)
        else:
            action = np.argmax(q_table[discrete_state])
        
        # 动作映射到力
        F_max = 2 * (config.M + config.m) * config.g / config.M
        F = (action - 7) * F_max / 7
        
        # 状态更新
        sol = solve_ivp(
            lambda t, y: dynamics(t, y, F),
            [0, config.dt],
            state,
            t_eval=[config.dt]
        )
        next_state = sol.y[:, -1]
        
        # 状态边界裁剪
        next_state[0] = np.clip(next_state[0], -config.max_position, config.max_position)
        next_state[1] = np.clip(next_state[1], -config.max_velocity, config.max_velocity)
        next_state[2] = np.clip(next_state[2], -config.max_angle, config.max_angle)
        next_state[3] = np.clip(next_state[3], -config.max_angular_vel, config.max_angular_vel)
        
        # 计算奖励
        reward = reward_function(state, action, next_state)
        total_reward += reward
        
        # Q学习更新
        next_discrete_state = discretize_state(next_state)
        max_next_q = np.max(q_table[next_discrete_state])
        
        td_target = reward + config.gamma * max_next_q
        td_error = td_target - q_table[discrete_state + (action,)]
        q_table[discrete_state + (action,)] += alpha * td_error
        
        # 记录数据
        datatrans.append([next_state[2], next_state[3], next_state[0], next_state[1]])
        
        # 更新状态
        state = next_state
        
        # 终止条件
        if abs(state[2]) > np.pi / 6 or abs(state[0]) > config.max_position:
            break
    
    # 参数衰减
    epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
    alpha = max(config.alpha_min, alpha * config.alpha_decay)
    
    # 记录统计
    episode_rewards.append(total_reward)
    episode_steps.append(step + 1)
    
    # 保存数据
    if (episode + 1) % 10 == 0:
        np.savetxt(
            f"pendulum_data_v4_run_{episode+1}.csv",
            datatrans,
            delimiter=",",
            header="Angle(rad),AngularVel(rad/s),Position(m),Velocity(m/s)",
            comments=""
        )
        np.save("q_table_v4.npy", q_table)
        
        # 打印统计信息
        avg_reward = np.mean(episode_rewards[-10:])
        avg_steps = np.mean(episode_steps[-10:])
        print(f"Episode {episode+1:3d} | "
              f"Steps: {step+1:3d} | "
              f"Reward: {total_reward:7.2f} | "
              f"Avg10: {avg_reward:7.2f} | "
              f"ε: {epsilon:.3f} | "
              f"α: {alpha:.3f}")

print("-" * 60)
print("训练完成！")
print(f"最终平均奖励（最后10回合）: {np.mean(episode_rewards[-10:]):.2f}")
print(f"最终平均步数（最后10回合）: {np.mean(episode_steps[-10:]):.2f}")

# 保存训练统计
np.savetxt(
    "training_stats_v4.csv",
    np.column_stack([episode_rewards, episode_steps]),
    delimiter=",",
    header="TotalReward,Steps",
    comments=""
)
