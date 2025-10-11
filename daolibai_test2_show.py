import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ✅ 设置中文显示（适用于支持中文的终端）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 路径设置
model_path = "trained_model.keras"
reward_path = "rewards_per_episode.npy"
sim_path = "simulated_states.npy"

# ✅ 加载模型
try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 成功加载模型：{model_path}")
except Exception as e:
    print(f"❌ 加载模型失败：{e}")
    model = None

# ✅ 加载奖励数据
try:
    rewards = np.load(reward_path)
    print(f"✅ 成功加载奖励数据，共 {len(rewards)} 个回合")
except Exception as e:
    print(f"❌ 加载奖励数据失败：{e}")
    rewards = []

# ✅ 奖励曲线可视化
if len(rewards) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='每回合奖励')

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    if len(rewards) >= 50:
        avg_rewards = moving_average(rewards, 50)
        plt.plot(range(49, len(rewards)), avg_rewards, label='滑动平均 (50)', linestyle='--')

    plt.title("训练过程中的总奖励变化")
    plt.xlabel("训练回合 Episode")
    plt.ylabel("总奖励 Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ 模型预测示例
if model is not None:
    sample_state = np.array([[0.0, 0.0, 0.05, 0.0]])
    q_values = model.predict(sample_state)
    action = np.argmax(q_values)
    action_map = {0: "← 施加负力", 1: "⏸ 无操作", 2: "→ 施加正力"}

    print(f"🎯 示例状态: {sample_state.flatten()}")
    print(f"🤖 预测的 Q 值: {q_values}")
    print(f"🎬 推荐动作: {action} ({action_map[action]})")

# ✅ 倒立摆模拟运行动画（基于保存的状态序列）
try:
    sim_states = np.load(sim_path)
    print(f"✅ 成功加载模拟状态数据，共 {len(sim_states)} 帧")

    # 绘图函数
    def animate(i):
        plt.cla()
        x, _, theta, _ = sim_states[i]
        cart_y = 0
        pole_len = 1.0

        # 小车
        plt.plot([x - 0.2, x + 0.2], [cart_y, cart_y], 'k', linewidth=8)

        # 摆杆
        pole_x = x + pole_len * np.sin(theta)
        pole_y = cart_y + pole_len * np.cos(theta)
        plt.plot([x, pole_x], [cart_y, pole_y], 'r-', linewidth=3)
        plt.plot(pole_x, pole_y, 'bo', markersize=8)

        plt.xlim(-50, 50)
        plt.ylim(-10, 10)
        plt.title(f"倒立摆模拟运行 - 第 {i+1} 帧")
        plt.grid(True)

    fig = plt.figure(figsize=(6, 4))
    ani = animation.FuncAnimation(fig, animate, frames=len(sim_states), interval=50)
    plt.show()

except Exception as e:
    print(f"❌ 加载或显示模拟状态失败：{e}")
