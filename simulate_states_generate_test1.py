import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

# 复制 DQNModel 定义（必须和训练时一致）
class DQNModel(tf.keras.Model):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense3 = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    # 通过 get_config 返回模型配置
    def get_config(self):
        config = super(DQNModel, self).get_config()
        return config
    
    # 通过配置恢复模型
    @classmethod
    def from_config(cls, config):
        return cls()

# 注册自定义对象
get_custom_objects()['DQNModel'] = DQNModel

# 加载模型
model = load_model("trained_model.keras", compile=False)

# -----------------------------
# 使用动力学函数模拟
# -----------------------------
import math
import random

M = 1.0
m = 0.1
l = 1.0
g = 9.81
b = 0.05

def dynamics(state, F):
    x, x_dot, theta, theta_dot = state
    I = (1 / 3) * m * (2 * l)**2
    theta = np.clip(theta, -math.pi, math.pi)
    theta_dot = np.clip(theta_dot, -10, 10)
    denom = M + m * (1 - np.cos(theta)**2)
    if np.abs(denom) < 1e-8:
        denom = 1e-8
    xddot = (F + m * l * theta_dot**2 * np.sin(theta) - m * g * np.sin(theta) * np.cos(theta)) / denom
    wdot = (-m * l * xddot * np.cos(theta) + (m + M) * g * np.sin(theta)) / (I + m * l**2)
    
    # 限制小车速度 (10单位/秒)
    new_x_dot = x_dot + xddot
    new_x_dot = np.clip(new_x_dot, -10, 10)  # 限制速度

    # 更新状态
    new_x = x + new_x_dot
    new_theta = theta + theta_dot
    new_theta_dot = theta_dot + wdot
    
    # 限制小车位置在[-50, 50]范围内
    new_x = np.clip(new_x, -50, 50)
    
    return [new_x, new_x_dot, new_theta, new_theta_dot]

# 初始化状态
state = [random.uniform(-0.05, 0.05), 0.0, random.uniform(-0.05, 0.05), 0.0]

trajectory = []
for t in range(1200):
    state_input = np.array(state).reshape(1, -1)
    q_values = model(state_input).numpy()
    action = np.argmax(q_values)
    F = [-10, 0, 10][action]
    next_state = dynamics(state, F)
    trajectory.append(next_state)
    state = next_state
    print(t)

trajectory = np.array(trajectory)
np.save("simulated_states.npy", trajectory)

print(f"✅ 已生成 simulated_states.npy，共 {len(trajectory)} 帧")
