import numpy as np
import tensorflow as tf
from collections import deque
import random
import math
import matplotlib.pyplot as plt
import keyboard  # 用于按键监听，实时调整 epsilon

# 动力学参数
M = 1.0  # 小车的质量
m = 0.1  # 摆杆的质量
l = 1.0  # 摆杆的长度
g = 9.81  # 重力加速度
b = 0.05  # 阻力系数（暂未使用）

# amazonq-ignore-next-line
# DQN 参数
alpha = 0.001  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 探索率，初始为 1，逐渐衰减
epsilon_min = 0.01  # 最小探索率
epsilon_decay = 0.995  # 每回合衰减的探索率
batch_size = 64  # 批次大小
memory_size = 10000  # 经验回放的最大大小
num_episodes = 1000  # 训练的回合数
target_update = 200  # 每 200 回合更新目标网络

# 经验回放的缓冲区
memory = deque(maxlen=memory_size)  # 用于存储经验元组（state, action, reward, next_state）

# DQN 网络模型
class DQNModel(tf.keras.Model):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')  # 第一层全连接层
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')  # 第二层全连接层
        self.dense3 = tf.keras.layers.Dense(3)  # 输出层，3 个动作（加速、减速、不变）

    def call(self, inputs):
        x = self.dense1(inputs)  # 通过第一层
        x = self.dense2(x)  # 通过第二层
        return self.dense3(x)  # 输出 Q 值

    def get_config(self):
        config = super(DQNModel, self).get_config()
        return config

    # amazonq-ignore-next-line
    @classmethod
    def from_config(cls, config):
        return cls()

# 初始化 Q 网络和目标网络
# amazonq-ignore-next-line
q_network = DQNModel()  # 当前 Q 网络
target_network = DQNModel()  # 目标 Q 网络
# 权重复制将在模型首次使用后进行

# 动力学方程
def dynamics(state, F):
    """根据小车的状态和作用力计算下一状态"""
    x, x_dot, theta, theta_dot = state  # 解包当前状态
    I = m * l**2  # 点质量倒立摆的转动惯量
    theta = np.clip(theta, -math.pi, math.pi)  # 限制摆杆角度在 -pi 到 pi 范围内
    # amazonq-ignore-next-line
    theta_dot = np.clip(theta_dot, -10, 10)  # 限制摆杆角速度

    # 动力学方程，计算加速度和角加速度
    denom = M + m * (1 - np.cos(theta)**2)
    if np.abs(denom) < 1e-8:
        # amazonq-ignore-next-line
        denom = 1e-8  # 防止除零
    xddot = (F + m * l * theta_dot**2 * np.sin(theta) - m * g * np.sin(theta) * np.cos(theta)) / denom
    wdot = (-m * l * xddot * np.cos(theta) + (m + M) * g * np.sin(theta)) / (I + m * l**2)
# amazonq-ignore-next-line

    # 更新小车的速度和位置，限制在合理范围内
    new_x_dot = x_dot + xddot
    new_x_dot = np.clip(new_x_dot, -10, 10)  # 限制小车速度
    new_x = x + new_x_dot  # 更新小车位置
    new_theta = theta + theta_dot  # 更新摆杆角度
    new_theta_dot = theta_dot + wdot  # 更新摆杆角速度

    # 限制小车位置在[-50, 50]范围内
    new_x = np.clip(new_x, -50, 50)

    return [new_x, new_x_dot, new_theta, new_theta_dot]

# 离散化状态
def discretize_state(state):
    """将连续状态空间离散化"""
    x, x_dot, theta, theta_dot = state
    theta_idx = np.digitize(theta, np.linspace(-math.pi, math.pi, 50)) - 1  # 将角度离散化
    theta_dot_idx = np.digitize(theta_dot, np.linspace(-10, 10, 50)) - 1  # 将角速度离散化
    x_idx = np.digitize(x, np.linspace(-2.4, 2.4, 50)) - 1  # 将位置离散化
    x_dot_idx = np.digitize(x_dot, np.linspace(-2.0, 2.0, 50)) - 1  # 将速度离散化
    return (theta_idx, theta_dot_idx, x_idx, x_dot_idx)

def get_reward(theta, theta_dot, x, x_dot, stable_time, max_stable_time):
    """计算奖励值"""
    reward = 0

    # 1. 惩罚摆杆的角度
    reward -= abs(theta) * 10  # 摆杆角度越大，奖励越低

    # 2. 边缘停留惩罚：小车位置接近边缘时的惩罚
    if abs(x) > 45:  # 如果小车接近边缘位置（[-50, 50] 范围内），给与惩罚
        reward -= 20  # 边缘惩罚，可以根据需求调整惩罚的强度

    # 3. 摆杆稳定时间奖励加成：当摆杆接近竖直位置时，累计稳定时间奖励
    if abs(theta) < 0.1:  # 摆杆接近竖直
        stable_reward = (stable_time / max_stable_time) * 10  # 按比例增加奖励
        reward += stable_reward  # 累计奖励加成

    # 4. 惩罚小车位置
    reward -= 0.1 * abs(x)  # 惩罚小车位置

    # 5. 惩罚小车速度
    reward -= 0.1 * abs(x_dot)  # 惩罚小车速度

    # 6. 如果摆杆非常接近竖直位置时，给更高的奖励
    if abs(theta) < 0.05:  # 摆杆更接近竖直
        reward += 50  # 可以根据需要调整这部分奖励

    # amazonq-ignore-next-line
    # amazonq-ignore-next-line
    # amazonq-ignore-next-line
    # amazonq-ignore-next-line
    return reward


# 训练 DQN
def train():
    global epsilon
    rewards_per_episode = []  # 用于记录每一回合的奖励
    target_initialized = False  # 目标网络初始化标志
    for episode in range(num_episodes):
        # 初始化状态 [小车位置, 小车速度, 摆杆角度, 摆杆角速度]
        state = [random.uniform(-2.4, 2.4), random.uniform(-2, 2), random.uniform(-math.pi, math.pi), random.uniform(-10, 10)]  
        # amazonq-ignore-next-line
        total_reward = 0
        stable_time = 0  # 摆杆稳定时间计数器
        max_stable_time = 500  # 最大稳定时间限制
        
        for t in range(200):  # 每回合最大 200 步
            state_input = np.array(state).reshape(1, -1)  # 将状态转化为模型输入格式
            
            # amazonq-ignore-next-line
            # 按键检查（实时调整 epsilon 值）
            if keyboard.is_pressed('up'):
                epsilon = min(epsilon + 0.05, 1.0)
                print("epsilon = "+ epsilon)
            if keyboard.is_pressed('down'):
                epsilon = max(epsilon - 0.05, epsilon_min)
                print("epsilon = "+ epsilon)
                
            # epsilon-greedy 策略
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1, 2])  # 随机选择动作
            else:
                q_values = q_network(state_input)  # 使用 Q 网络预测 Q 值
                action = np.argmax(q_values)  # 选择最大 Q 值对应的动作
                
            # 初始化目标网络（在模型首次使用后）
            if not target_initialized:
                target_network.set_weights(q_network.get_weights())
                target_initialized = True
                
            F = [-10, 0, 10][action]  # 离散化的动作：加速、减速、不变
            
            # 动力学更新
            # amazonq-ignore-next-line
            new_state = dynamics(state, F)
            
            # 奖励计算
            reward = get_reward(new_state[2], new_state[3], new_state[0], new_state[1], stable_time, max_stable_time)
            # amazonq-ignore-next-line
            total_reward += reward
            # amazonq-ignore-next-line
            
            # 摆杆稳定时间加成
            if abs(new_state[2]) < 0.1:  # 摆杆接近竖直
                stable_time += 1
            else:
                stable_time = 0  # 如果摆杆不再稳定，重置时间
            
            # 存储经验
            memory.append((state, action, reward, new_state))
            state = new_state  # 更新状态
            
            # 更新网络
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states = zip(*batch)
                
                states = np.array(states)
                next_states = np.array(next_states)
                
                with tf.GradientTape() as tape:
                    q_values = q_network(states)
                    next_q_values = target_network(next_states)
                    target_q_values = rewards + gamma * np.max(next_q_values, axis=1)  # Q 学习目标
                    
                    action_q_values = tf.reduce_sum(q_values * tf.one_hot(actions, 3), axis=1)
                    loss = tf.reduce_mean(tf.square(target_q_values - action_q_values))  # 损失函数
                
                grads = tape.gradient(loss, q_network.trainable_variables)  # 计算梯度
                optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)  # 优化器
                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))  # 应用梯度更新参数
        
        # 每 200 回合更新目标网络
        if episode % target_update == 0:
            target_network.set_weights(q_network.get_weights())
        
        # 衰减 epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        rewards_per_episode.append(total_reward)  # 记录回合奖励
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
    
    # 保存模型和奖励记录
    q_network.save("trained_model.keras")
    np.save("rewards_per_episode.npy", rewards_per_episode)
    
    # 绘制奖励曲线
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.show()

# 训练 DQN 控制器
train()
