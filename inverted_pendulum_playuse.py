import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# 参数
M = 1.0   # 车质量
m = 0.1   # 杆质量
l = 1.0   # 摆长
g = 9.81  # 重力加速度

# 状态变量 [x v theta omega]
state = np.array([0.0, 0.0, 5 * np.pi / 180, 0.0])  # 5度初始角度

# 控制力
F = 0

datatrans = []

# 动力学方程
def dynamics(t, s, F):
    x, v, th, w = s

    # 系统参数
    b = 0.05  # 小车阻力（加入受力分析）
    I = (1/3) * m * (2 * l)**2  # 杆关于端点的转动惯量，注意l为质心到端点的距离
    
    # 用符号表达写出方程组
    # 设 xddot, wdot（即 θ̈）为未知数
    def equations(vars):
        xddot, wdot = vars

        # 杆质心加速度
        x_gc_ddot = xddot + l * wdot * np.cos(th) - l * w**2 * np.sin(th)
        y_gc_ddot = l * wdot * np.sin(th) + l * w**2 * np.cos(th)

        # 水平作用力 N（N_GC）
        N = m * x_gc_ddot

        # 竖直支持力 P（P_GC）
        P = m * g + m * y_gc_ddot

        # 小车水平力平衡：F = M * xddot + N + b * v
        eq1 = F - M * xddot - N - b * v

        # 杆的转动平衡方程：N * cosθ - P * sinθ = I * wdot / l
        eq2 = N * np.cos(th) - P * np.sin(th) - I * wdot / (2 * l)

        return [eq1, eq2]

    from scipy.optimize import fsolve
    xddot, wdot = fsolve(equations, [0, 0])

    return [v, xddot, w, wdot]

# 初始化图形
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

# 键盘事件处理
def on_press(event):
    global F
    if event.key == 'right':
        F = 10  # 右力
    elif event.key == 'left':
        F = -10  # 左力

def on_release(event):
    global F
    if event.key in ['right', 'left']:
        F = 0  # 释放时归零力

fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('key_release_event', on_release)

# 动画更新函数
def update(frame):
    global state, datatrans
    sol = solve_ivp(lambda t, y: dynamics(t, y, F), [0, 0.02], state, t_eval=[0.02])
    state = sol.y[:, -1]

    x = state[0]
    theta = state[2]
    omega = state[3]

    # 保存角度和角速度到 datatrans
    datatrans.append([theta, omega])

    # 更新小车位置
    cart.set_xy((x - cartW / 2, 0))

    # 允许摆杆 360° 范围旋转（使用 sinθ, cosθ 计算完整端点）
    rod_x = [x, x + 2 * l * np.sin(theta)]
    rod_y = [cartH, cartH - 2 * l * np.cos(theta)]   # 注意这里减号，保证 θ=0 时杆竖直向上

    rod.set_data(rod_x, rod_y)

    return cart, rod

# 创建动画
ani = FuncAnimation(fig, update, frames=200, interval=20, blit=True)

plt.show()

# 保存为 CSV 文件（可选）
df = np.DataFrame(datatrans, columns=["Angle (rad)", "Angular Velocity (rad/s)"])
df.to_csv("pendulum_data.csv", index=False)
