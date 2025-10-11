import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 加载训练数据
run_number = int(input("Enter the run number: "))  # 假设你要查看第1次训练的数据
data_file = f"pendulum_data_run_{run_number}.csv"
df = pd.read_csv(data_file)

# ---- 设置画布 ----
fig, ax = plt.subplots()
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.grid(True)

cartW = 0.4
cartH = 0.2
cart = plt.Rectangle((-cartW / 2, 0), cartW, cartH, facecolor=(0, 0.6, 0.8))
rod, = ax.plot([], [], lw=3, color='r')

ax.add_patch(cart)

# 动力学参数
M = 1.0   # 车质量
m = 0.1   # 杆质量
l = 1.0   # 摆长

# ---- 更新动画的函数 ----
def update(frame):
    # 获取当前状态（角度和角速度）
    theta = df.iloc[frame]["Angle (rad)"]
    omega = df.iloc[frame]["Angular Velocity (rad/s)"]
    
    # 更新小车位置
    x = 2 * l * np.sin(theta)
    cart.set_xy((x - cartW / 2, 0))

    # 更新摆杆位置
    rod_x = [x, x + 2 * l * np.sin(theta)]
    rod_y = [cartH, cartH - 2 * l * np.cos(theta)]
    rod.set_data(rod_x, rod_y)

    return cart, rod

# 创建动画
ani = FuncAnimation(fig, update, frames=len(df), interval=20, blit=True)

plt.show()
