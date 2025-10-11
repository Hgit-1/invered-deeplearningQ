import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 训练统计可视化 ====================
def plot_training_stats():
    """绘制训练过程统计图"""
    if not os.path.exists("training_stats_v4.csv"):
        print("未找到 training_stats_v4.csv 文件")
        return
    
    data = np.loadtxt("training_stats_v4.csv", delimiter=",", skiprows=1)
    episodes = np.arange(1, len(data) + 1)
    rewards = data[:, 0]
    steps = data[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 奖励曲线
    ax1.plot(episodes, rewards, 'b-', alpha=0.3, label='每回合奖励')
    # 移动平均
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}回合移动平均')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('回合数', fontsize=12)
    ax1.set_ylabel('累计奖励', fontsize=12)
    ax1.set_title('训练奖励曲线', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 步数曲线
    ax2.plot(episodes, steps, 'g-', alpha=0.3, label='每回合步数')
    if len(steps) >= window:
        moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], moving_avg_steps, 'orange', linewidth=2, label=f'{window}回合移动平均')
    ax2.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='最大步数')
    ax2.set_xlabel('回合数', fontsize=12)
    ax2.set_ylabel('步数', fontsize=12)
    ax2.set_title('训练步数曲线', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_stats_v4.png', dpi=300, bbox_inches='tight')
    print("✓ 训练统计图已保存: training_stats_v4.png")
    plt.show()

# ==================== 2. 单回合轨迹可视化 ====================
def plot_episode_trajectory(episode_num):
    """绘制指定回合的状态轨迹"""
    filename = f"pendulum_data_v4_run_{episode_num}.csv"
    if not os.path.exists(filename):
        print(f"未找到文件: {filename}")
        return
    
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    time = np.arange(len(data)) * 0.02  # 时间步长0.02秒
    
    angle = data[:, 0] * 180 / np.pi  # 转换为度数
    angular_vel = data[:, 1]
    position = data[:, 2]
    velocity = data[:, 3]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 角度
    axes[0, 0].plot(time, angle, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='失败边界')
    axes[0, 0].axhline(y=-30, color='orange', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('时间 (s)', fontsize=11)
    axes[0, 0].set_ylabel('角度 (°)', fontsize=11)
    axes[0, 0].set_title('摆杆角度', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 角速度
    axes[0, 1].plot(time, angular_vel, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('时间 (s)', fontsize=11)
    axes[0, 1].set_ylabel('角速度 (rad/s)', fontsize=11)
    axes[0, 1].set_title('摆杆角速度', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 位置
    axes[1, 0].plot(time, position, 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='边界')
    axes[1, 0].axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('时间 (s)', fontsize=11)
    axes[1, 0].set_ylabel('位置 (m)', fontsize=11)
    axes[1, 0].set_title('小车位置', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 速度
    axes[1, 1].plot(time, velocity, 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('时间 (s)', fontsize=11)
    axes[1, 1].set_ylabel('速度 (m/s)', fontsize=11)
    axes[1, 1].set_title('小车速度', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'第 {episode_num} 回合状态轨迹 (共{len(data)}步)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'trajectory_v4_run_{episode_num}.png', dpi=300, bbox_inches='tight')
    print(f"✓ 轨迹图已保存: trajectory_v4_run_{episode_num}.png")
    plt.show()

# ==================== 3. 相位图 ====================
def plot_phase_diagram(episode_num):
    """绘制相位图"""
    filename = f"pendulum_data_v4_run_{episode_num}.csv"
    if not os.path.exists(filename):
        print(f"未找到文件: {filename}")
        return
    
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    angle = data[:, 0] * 180 / np.pi
    angular_vel = data[:, 1]
    position = data[:, 2]
    velocity = data[:, 3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 摆杆相位图
    scatter1 = ax1.scatter(angle, angular_vel, c=np.arange(len(angle)), 
                          cmap='viridis', s=20, alpha=0.6)
    ax1.plot(angle, angular_vel, 'b-', alpha=0.3, linewidth=1)
    ax1.plot(angle[0], angular_vel[0], 'go', markersize=10, label='起点')
    ax1.plot(angle[-1], angular_vel[-1], 'ro', markersize=10, label='终点')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('角度 (°)', fontsize=12)
    ax1.set_ylabel('角速度 (rad/s)', fontsize=12)
    ax1.set_title('摆杆相位图', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='时间步')
    
    # 小车相位图
    scatter2 = ax2.scatter(position, velocity, c=np.arange(len(position)), 
                          cmap='plasma', s=20, alpha=0.6)
    ax2.plot(position, velocity, 'r-', alpha=0.3, linewidth=1)
    ax2.plot(position[0], velocity[0], 'go', markersize=10, label='起点')
    ax2.plot(position[-1], velocity[-1], 'ro', markersize=10, label='终点')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('位置 (m)', fontsize=12)
    ax2.set_ylabel('速度 (m/s)', fontsize=12)
    ax2.set_title('小车相位图', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='时间步')
    
    fig.suptitle(f'第 {episode_num} 回合相位图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'phase_diagram_v4_run_{episode_num}.png', dpi=300, bbox_inches='tight')
    print(f"✓ 相位图已保存: phase_diagram_v4_run_{episode_num}.png")
    plt.show()

# ==================== 4. 动画演示 ====================
def create_animation(episode_num):
    """创建倒立摆动画"""
    filename = f"pendulum_data_v4_run_{episode_num}.csv"
    if not os.path.exists(filename):
        print(f"未找到文件: {filename}")
        return
    
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    angle = data[:, 0]
    position = data[:, 2]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=2)
    
    # 绘制元素
    cart, = ax.plot([], [], 's', markersize=30, color='blue', label='小车')
    pole, = ax.plot([], [], 'o-', linewidth=4, markersize=10, color='red', label='摆杆')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    angle_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
    pos_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('位置 (m)', fontsize=12)
    ax.set_ylabel('高度 (m)', fontsize=12)
    ax.set_title(f'倒立摆动画 - 第 {episode_num} 回合', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    def init():
        cart.set_data([], [])
        pole.set_data([], [])
        time_text.set_text('')
        angle_text.set_text('')
        pos_text.set_text('')
        return cart, pole, time_text, angle_text, pos_text
    
    def animate(i):
        # 小车位置
        x = position[i]
        cart.set_data([x], [0])
        
        # 摆杆位置（长度2m）
        pole_x = [x, x + 2 * np.sin(angle[i])]
        pole_y = [0, 2 * np.cos(angle[i])]
        pole.set_data(pole_x, pole_y)
        
        # 文本信息
        time_text.set_text(f'时间: {i*0.02:.2f}s')
        angle_text.set_text(f'角度: {angle[i]*180/np.pi:.1f}°')
        pos_text.set_text(f'位置: {x:.2f}m')
        
        return cart, pole, time_text, angle_text, pos_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(data),
                        interval=20, blit=True, repeat=True)
    
    # 保存动画
    try:
        anim.save(f'pendulum_animation_v4_run_{episode_num}.gif', 
                 writer='pillow', fps=50, dpi=100)
        print(f"✓ 动画已保存: pendulum_animation_v4_run_{episode_num}.gif")
    except Exception as e:
        print(f"保存动画失败: {e}")
    
    plt.show()

# ==================== 5. 多回合对比 ====================
def compare_episodes():
    """对比多个回合的性能"""
    csv_files = sorted(glob.glob("pendulum_data_v4_run_*.csv"))
    if not csv_files:
        print("未找到任何CSV文件")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for filename in csv_files:
        episode_num = int(filename.split('_')[-1].split('.')[0])
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        time = np.arange(len(data)) * 0.02
        
        angle = data[:, 0] * 180 / np.pi
        angular_vel = data[:, 1]
        position = data[:, 2]
        velocity = data[:, 3]
        
        label = f'回合 {episode_num}'
        
        axes[0, 0].plot(time, angle, label=label, alpha=0.7)
        axes[0, 1].plot(time, angular_vel, label=label, alpha=0.7)
        axes[1, 0].plot(time, position, label=label, alpha=0.7)
        axes[1, 1].plot(time, velocity, label=label, alpha=0.7)
    
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('角度 (°)')
    axes[0, 0].set_title('角度对比', fontweight='bold')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('角速度 (rad/s)')
    axes[0, 1].set_title('角速度对比', fontweight='bold')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('位置 (m)')
    axes[1, 0].set_title('位置对比', fontweight='bold')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('速度 (m/s)')
    axes[1, 1].set_title('速度对比', fontweight='bold')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('多回合性能对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('episodes_comparison_v4.png', dpi=300, bbox_inches='tight')
    print("✓ 对比图已保存: episodes_comparison_v4.png")
    plt.show()

# ==================== 主菜单 ====================
def main():
    print("=" * 60)
    print("倒立摆 Q学习 V4 数据可视化工具")
    print("=" * 60)
    
    while True:
        print("\n请选择功能:")
        print("1. 训练统计图 (奖励和步数曲线)")
        print("2. 单回合轨迹图")
        print("3. 单回合相位图")
        print("4. 单回合动画演示")
        print("5. 多回合对比图")
        print("6. 生成所有图表")
        print("0. 退出")
        
        choice = input("\n输入选项 (0-6): ").strip()
        
        if choice == '1':
            plot_training_stats()
        
        elif choice == '2':
            episode = input("输入回合数 (10, 20, 30...): ").strip()
            try:
                plot_episode_trajectory(int(episode))
            except ValueError:
                print("无效输入，请输入数字")
        
        elif choice == '3':
            episode = input("输入回合数 (10, 20, 30...): ").strip()
            try:
                plot_phase_diagram(int(episode))
            except ValueError:
                print("无效输入，请输入数字")
        
        elif choice == '4':
            episode = input("输入回合数 (10, 20, 30...): ").strip()
            try:
                create_animation(int(episode))
            except ValueError:
                print("无效输入，请输入数字")
        
        elif choice == '5':
            compare_episodes()
        
        elif choice == '6':
            print("\n生成所有图表...")
            plot_training_stats()
            
            # 查找所有CSV文件
            csv_files = sorted(glob.glob("pendulum_data_v4_run_*.csv"))
            if csv_files:
                # 生成第一个和最后一个回合的详细图
                first_episode = int(csv_files[0].split('_')[-1].split('.')[0])
                last_episode = int(csv_files[-1].split('_')[-1].split('.')[0])
                
                print(f"\n生成第 {first_episode} 回合图表...")
                plot_episode_trajectory(first_episode)
                plot_phase_diagram(first_episode)
                
                print(f"\n生成第 {last_episode} 回合图表...")
                plot_episode_trajectory(last_episode)
                plot_phase_diagram(last_episode)
                
                compare_episodes()
                print("\n✓ 所有图表生成完成！")
            else:
                print("未找到CSV文件")
        
        elif choice == '0':
            print("退出程序")
            break
        
        else:
            print("无效选项，请重新输入")

if __name__ == "__main__":
    main()
