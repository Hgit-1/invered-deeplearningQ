import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取当前目录和可用的CSV文件
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(current_dir, "pendulum_data_run_*.csv"))

if not csv_files:
    print("错误：未找到任何CSV文件")
    print(f"当前目录: {current_dir}")
    exit(1)

# 提取可用的回合数
available_runs = []
for f in csv_files:
    basename = os.path.basename(f)
    try:
        num = int(basename.split('_')[-1].replace('.csv', '').replace('interrupted', ''))
        available_runs.append(num)
    except:
        pass

available_runs.sort()

print("="*60)
print(f"找到 {len(available_runs)} 个训练数据文件")
print(f"可用回合数: {available_runs[:10]}{'...' if len(available_runs) > 10 else ''}")
print(f"最小回合: {min(available_runs)}, 最大回合: {max(available_runs)}")
print("="*60)

# 加载训练数据
while True:
    try:
        run_number = int(input("\n输入回合数 (或输入0退出): "))
        if run_number == 0:
            print("退出程序")
            exit(0)
        
        # 尝试多种文件名格式
        possible_files = [
            os.path.join(current_dir, f"pendulum_data_run_{run_number}.csv"),
            os.path.join(current_dir, f"pendulum_data_run_{run_number}_interrupted.csv"),
            f"pendulum_data_run_{run_number}.csv",
            f"pendulum_data_run_{run_number}_interrupted.csv"
        ]
        
        data_file = None
        for f in possible_files:
            if os.path.exists(f):
                data_file = f
                break
        
        if data_file is None:
            print(f"\n错误：未找到回合 {run_number} 的数据文件")
            print(f"请输入以下可用回合数之一: {available_runs[:20]}")
            continue
        
        print(f"\n加载文件: {os.path.basename(data_file)}")
        df = pd.read_csv(data_file)
        print(f"数据点数: {len(df)}")
        break
        
    except ValueError:
        print("\n错误：请输入有效的数字")
    except Exception as e:
        print(f"\n错误：{e}")

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

# 添加时间文本
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- 更新动画的函数 ----
def update(frame):
    # 获取当前状态（角度和角速度）
    theta = df.iloc[frame]["Angle (rad)"]
    omega = df.iloc[frame]["Angular Velocity (rad/s)"]
    
    # 更新时间文本
    time_text.set_text(f'Step: {frame+1}/{len(df)}\nAngle: {theta*180/np.pi:.1f}deg\nAngular Vel: {omega:.2f} rad/s')
    
    # 更新小车位置
    x = 2 * l * np.sin(theta)
    cart.set_xy((x - cartW / 2, 0))

    # 更新摆杆位置
    rod_x = [x, x + 2 * l * np.sin(theta)]
    rod_y = [cartH, cartH - 2 * l * np.cos(theta)]
    rod.set_data(rod_x, rod_y)

    return cart, rod, time_text

# 添加标题
ax.set_title(f'Inverted Pendulum - Episode {run_number} ({len(df)} steps)', fontsize=14, fontweight='bold')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Height (m)')

# 创建动画
print("\n生成动画中...")
ani = FuncAnimation(fig, update, frames=len(df), interval=20, blit=True)

print("显示动画窗口（关闭窗口结束程序）")
plt.show()
print("\n程序结束")
