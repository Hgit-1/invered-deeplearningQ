# 水质评分强化学习模型
import pandas as pd
import numpy as np
import random
import time

# 参数
epsilon = 0.5   # 探索概率
alpha = 0.1     # 学习率
gamma = 0.9     # 奖励递减值
target_score = 90  # 目标分数

# 加载实验数据
try:
    water_data = np.loadtxt('c:/Users/H2010/Documents/water_score/water_score.txt')
except:
    water_data = np.loadtxt('water_score.txt')
apertures = [0.1, 0.2, 0.5, 1, 5, 10]  # 孔径(um)
layers = [1, 2, 3, 4, 5, 6]  # 层数

# 状态定义：水质分数 0-100
states = range(101)

# 动作定义：(孔径索引, 层数索引)
actions = [(i, j) for i in range(6) for j in range(6)]

# 初始化Q值表
q_table = pd.DataFrame(data=[[0 for _ in range(len(actions))] for _ in states], 
                      index=states, columns=range(len(actions)))

def get_next_state(current_state, action):
    """根据实验数据计算下一个状态"""
    aperture_idx, layer_idx = action
    
    # 从实验数据获取过滤后分数
    filtered_score = water_data[aperture_idx, layer_idx]
    
    return int(filtered_score)

def calculate_reward(current_score, target_score):
    """计算奖励：-(当前分数 - 目标分数)^2"""
    return -(current_score - target_score) ** 2

def get_valid_actions(state):
    """获取当前状态下的有效动作"""
    return actions  # 所有动作都有效

# 训练历史记录
history = np.zeros((1000, 10))

print("开始训练水质评分模型...")

for episode in range(10):
    current_state = random.randint(20, 60)  # 随机初始分数
    total_steps = 0
    episode_reward = 0
    
    print(f"\n第 {episode + 1} 轮训练，初始状态: {current_state}")
    
    while total_steps < 50:  # 限制最大步数
        # 选择动作（ε-贪婪策略）
        if (random.uniform(0, 1) < epsilon) or ((q_table.loc[current_state] == 0).all()):
            current_action = random.choice(get_valid_actions(current_state))
        else:
            best_action_idx = q_table.loc[current_state].idxmax()
            current_action = actions[best_action_idx]
        
        # 执行动作，获得下一状态
        next_state = get_next_state(current_state, current_action)
        
        # 计算奖励
        reward = calculate_reward(next_state, target_score)
        episode_reward += reward
        
        # 获取下一状态的Q值
        next_state_q_values = q_table.loc[next_state]
        
        # 更新Q值
        action_index = actions.index(current_action)
        q_table.iloc[current_state, action_index] += alpha * (
            reward + gamma * next_state_q_values.max() - q_table.iloc[current_state, action_index]
        )
        
        current_state = next_state
        total_steps += 1
        history[total_steps, episode] = current_state
        
        # 如果达到目标分数附近，结束本轮
        if abs(current_state - target_score) <= 3:
            print(f"  达到目标！最终分数: {current_state}, 步数: {total_steps}")
            break
    
    print(f"  第 {episode + 1} 轮完成，总奖励: {episode_reward:.2f}, 最终分数: {current_state}")
    time.sleep(1)

print("\n训练完成！")
print("\nQ值表（前10行）:")
print(q_table.head(10))

# 保存Q值表
q_table.to_csv('c:/Users/H2010/Documents/water-score-aiana/q_table.csv')
print("Q值表已保存到 q_table.csv")

# 保存模型参数
model_data = {
    'water_data': water_data,
    'apertures': apertures,
    'layers': layers,
    'actions': actions,
    'target_score': target_score
}
np.savez('c:/Users/H2010/Documents/water-score-aiana/model.npz', **model_data)
print("模型已保存到 model.npz")

print(f"\n最佳策略示例（状态50）:")
best_action_idx = q_table.loc[50].idxmax()
best_action = actions[best_action_idx]
aperture_idx, layer_idx = best_action
print(f"最佳动作: 孔径={apertures[aperture_idx]}um, 层数={layers[layer_idx]}")
print(f"预期分数: {water_data[aperture_idx, layer_idx]}")