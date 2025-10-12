import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os
import time
from pynput import keyboard

# 参数
epsilon = 1.0
alpha = 0.02
gamma = 0.9
max_steps = 800  # 步数翻倍（原先200，现在800）

# 按键监听标志
stop_flag = False

def on_press(key):
    global stop_flag
    try:
        if key == keyboard.Key.up:  # 监听上箭头
            stop_flag = True
            print("\n[按键检测] 检测到上箭头按压，将在本次结束后停止...")
            return False  # 停止监听
    except:
        pass

# 启动按键监听线程
listener = keyboard.Listener(on_press=on_press)
listener.start()

# 创建保存目录
save_dir = "daolibai_test1V2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建保存目录: {save_dir}")

# 奖励函数
rewards = np.array([-abs(i - 180) for i in range(361)]) #！

# Q表初始化
q_table_path = os.path.join(save_dir, "q_table.npy")
q_table = np.load(q_table_path) if os.path.exists(q_table_path) else np.zeros((361, 15))

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

# 训练 2000 回合
print("="*70)
print("开始训练 - 按上箭头键可随时停止并保存当前回合数据")
print("="*70)

for run in range(10000):
    if stop_flag:
        print(f"\n[用户中断] 在第 {run+1} 回合前停止训练")
        break
    
    state = np.array([0.0, 0.0, 3, 0.0]) 
    datatrans = []

    step=0
    total_reward = 0
    start_time = time.time()
    
    while step < max_steps:  # 使用步数限制
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
        sol = solve_ivp(lambda t, y: dynamics(t, y, F), [0, 0.005], state, t_eval=[0.005])
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
        reward = -abs(state[2]) #-abs(state[3])
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
        
        # 检查按键中断
        if stop_flag:
            print(f"\n[按键中断] 在第 {run+1} 回合第 {step} 步检测到中断")
            break
        
        # Q 学习更新
        max_next_q = np.max(q_table[next_state])
        q_table[current_state, current_action] += alpha * (
            total_reward + gamma * max_next_q - q_table[current_state, current_action]
        )

    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    # 计算统计信息
    final_angle = state[2]
    final_angular_vel = state[3]
    final_position = state[0]
    final_velocity = state[1]
    avg_reward = total_reward / step if step > 0 else 0
    
    # 每次都输出详细信息
    print(f"\n{'='*70}")
    print(f"回合 {run+1:5d} 完成")
    print(f"{'-'*70}")
    print(f"  步数:         {step:4d} / {max_steps} 步")
    print(f"  运行时间:     {elapsed_time:.3f} 秒")
    print(f"  累计奖励:     {total_reward:8.2f}")
    print(f"  平均奖励:     {avg_reward:8.2f}")
    print(f"  最终角度:     {final_angle:7.2f}°")
    print(f"  最终角速度:   {final_angular_vel:7.2f} rad/s")
    print(f"  最终位置:     {final_position:7.2f} m")
    print(f"  最终速度:     {final_velocity:7.2f} m/s")
    print(f"  探索率 ε:     {epsilon:.4f}")
    
    # 判断终止原因
    if step >= max_steps:
        print(f"  终止原因:     达到最大步数 ✓")
    elif abs(state[0]) == 30:
        print(f"  终止原因:     位置超限 ✗")
    elif abs(state[1]) == 10:
        print(f"  终止原因:     速度超限 ✗")
    elif total_reward < -1000:
        print(f"  终止原因:     奖励过低 ✗")
    elif stop_flag:
        print(f"  终止原因:     用户中断 ⚠")
    else:
        print(f"  终止原因:     其他")
    print(f"{'='*70}")
    
    # 每50回合或按键中断时保存
    if (run + 1) % 50 == 0 or stop_flag:
        csv_path = os.path.join(save_dir, f"pendulum_data_run_{run+1}.csv")
        np.savetxt(csv_path, datatrans, delimiter=",", 
                   header="Angle (rad),Angular Velocity (rad/s)", comments="")
        np.save(q_table_path, q_table)
        print(f"\n[保存] 数据已保存: {csv_path}")
        print(f"[保存] Q表已保存: {q_table_path}\n")
    
    # 如果检测到按键中断，保存并退出
    if stop_flag:
        # 确保保存当前回合数据
        if (run + 1) % 50 != 0:  # 如果不是50的倍数，额外保存一次
            csv_interrupted_path = os.path.join(save_dir, f"pendulum_data_run_{run+1}.csv")
            np.savetxt(csv_interrupted_path, datatrans, delimiter=",", 
                       header="Angle (rad),Angular Velocity (rad/s)", comments="")
            print(f"[中断保存] 当前回合数据已保存: {csv_interrupted_path}")
        break

print("\n" + "="*70)
print("训练结束")
print("="*70)
listener.stop()
