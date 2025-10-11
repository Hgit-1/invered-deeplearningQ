# 水质评分模型调用程序
import pandas as pd
import numpy as np

class WaterScorePredictor:
    def __init__(self):
        # 加载训练好的Q值表和模型参数
        self.q_table = pd.read_csv('c:/Users/H2010/Documents/water-score-aiana/q_table.csv', index_col=0)
        model_data = np.load('c:/Users/H2010/Documents/water-score-aiana/model.npz', allow_pickle=True)
        
        self.water_data = model_data['water_data']
        self.apertures = model_data['apertures']
        self.layers = model_data['layers']
        self.actions = [(i, j) for i in range(6) for j in range(6)]  # 重建动作列表
        self.target_score = model_data['target_score']
    
    def predict_best_action(self, current_score):
        """根据当前分数预测最佳动作"""
        current_score = int(current_score)
        if current_score < 0 or current_score > 100:
            return None, "分数超出范围 (0-100)"
        
        # 获取当前状态的最佳动作
        best_action_idx = int(self.q_table.loc[current_score].idxmax())
        best_action = self.actions[best_action_idx]
        aperture_idx, layer_idx = best_action
        
        result = {
            'aperture': self.apertures[aperture_idx],
            'layers': self.layers[layer_idx],
            'expected_score': self.water_data[aperture_idx, layer_idx],
            'improvement': self.water_data[aperture_idx, layer_idx] - current_score
        }
        
        return result, "成功"
    
    def get_all_options(self, current_score):
        """获取所有可能的动作及其效果"""
        current_score = int(current_score)
        if current_score < 0 or current_score > 100:
            return None, "分数超出范围 (0-100)"
        
        options = []
        for i, action in enumerate(self.actions):
            aperture_idx, layer_idx = action
            q_value = self.q_table.iloc[current_score, i]
            expected_score = self.water_data[aperture_idx, layer_idx]
            
            options.append({
                'aperture': self.apertures[aperture_idx],
                'layers': self.layers[layer_idx],
                'expected_score': expected_score,
                'improvement': expected_score - current_score,
                'q_value': q_value
            })
        
        # 按Q值排序
        options.sort(key=lambda x: x['q_value'], reverse=True)
        return options, "成功"

def main():
    try:
        predictor = WaterScorePredictor()
        print("水质评分预测系统已加载")
        
        while True:
            print("\n=== 水质评分预测系统 ===")
            print("1. 获取最佳过滤方案")
            print("2. 查看所有方案排名")
            print("3. 退出")
            
            choice = input("请选择功能 (1-3): ")
            
            if choice == '1':
                try:
                    score = float(input("请输入当前水质分数 (0-100): "))
                    result, msg = predictor.predict_best_action(score)
                    
                    if result:
                        print(f"\n当前分数: {score}")
                        print(f"推荐方案:")
                        print(f"  孔径: {result['aperture']}um")
                        print(f"  层数: {result['layers']}")
                        print(f"  预期分数: {result['expected_score']}")
                        print(f"  分数提升: {result['improvement']:.1f}")
                    else:
                        print(f"错误: {msg}")
                except ValueError:
                    print("请输入有效数字")
            
            elif choice == '2':
                try:
                    score = float(input("请输入当前水质分数 (0-100): "))
                    options, msg = predictor.get_all_options(score)
                    
                    if options:
                        print(f"\n当前分数: {score}")
                        print("所有方案排名 (按推荐度排序):")
                        print("-" * 60)
                        for i, opt in enumerate(options[:10]):  # 显示前10个
                            print(f"{i+1:2d}. 孔径:{opt['aperture']:4}um 层数:{opt['layers']} "
                                  f"预期:{opt['expected_score']:2.0f} 提升:{opt['improvement']:+4.1f} "
                                  f"Q值:{opt['q_value']:6.2f}")
                    else:
                        print(f"错误: {msg}")
                except ValueError:
                    print("请输入有效数字")
            
            elif choice == '3':
                print("退出系统")
                break
            
            else:
                print("无效选择，请重新输入")
    
    except FileNotFoundError:
        print("错误: 找不到模型文件，请先运行训练程序")
    #except Exception as e:
    #    print(f"系统错误: {e}")

if __name__ == "__main__":
    main()