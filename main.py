import sys
import os
import torch
import json
import traceback
import gymnasium
import stable_baselines3
import numpy as np
#..
from modules.env import ThicknessOptimizationEnv, plot_reflection_comparison
from modules.material_optimizer import MaterialOptimizer
from modules.training import train_model, evaluate_model

def main():
    # 检查Python环境
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # 检查必要包的版本
    print(f"Gymnasium version: {gymnasium.__version__}")
    print(f"Stable-baselines3 version: {stable_baselines3.__version__}")
    
    try:
        # 检查CUDA是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建tensorboard日志目录
        os.makedirs("./thickness_optimization_tensorboard/", exist_ok=True)
        
        # 定义目标反射率
        target_reflection = {
            "wavelength_ranges": [
                {"range": [0.38e-6, 0.8e-6], "target": 0.0},
                {"range": [3e-6, 5e-6], "target": 1.0},
                {"range": [5e-6, 8e-6], "target": 0.0},
                {"range": [8e-6, 14e-6], "target": 1.0}
            ]
        }
        
        # 第一阶段：选择材料
        num_layers = 8  # 明确设置层数
        material_optimizer = MaterialOptimizer(target_reflection, num_layers=num_layers)
        best_materials = material_optimizer.optimize_materials()
        print("最优材料序列:", best_materials)
        
        # 确保最终的材料序列没有相邻相同材料
        merged_materials, _ = material_optimizer._merge_adjacent_same_materials(best_materials)
        if len(merged_materials) != len(best_materials):
            print(f"合并相邻相同材料后的序列: {merged_materials} (从{len(best_materials)}层减少到{len(merged_materials)}层)")
            best_materials = merged_materials
            num_layers = len(best_materials)  # 更新层数
        
        # 第二阶段：优化厚度和顺序
        env = ThicknessOptimizationEnv(
            target_reflection=target_reflection,
            materials_sequence=best_materials,
            num_layers=num_layers  # 使用更新后的层数
        )
        
        # 继续使用现有的训练逻辑
        best_model, env, config = train_model(env)
        
        print("模型训练完成")
        if best_model is not None:
            print(f"best_model loaded successfully-------")
        print("\n开始最终评估...")
        # 使用最优模型进行评估
        best_mse, best_reward, best_thicknesses = evaluate_model(best_model, env, num_episodes=10, render=False)
        
        print("\n最终配置:")
        print(f"最佳奖励值: {best_reward}")
        print(f"最佳MSE: {best_mse}")
        material_names = list(env.available_materials.keys())
        print("最佳厚度配置 (nm):")
        for i, t in enumerate(best_thicknesses):
            print(f"Layer {i+1} ({material_names[env.fixed_materials[i]]}):"
                  f" {t*1e9} nm")
        
        # 更新最佳厚度
        with open('model_config.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        if best_mse < data['best_mse']:
        # 将 NumPy 数组转换为列表
            data['best_thicknesses'] = best_thicknesses.tolist()  # 添加 .tolist()
            data['best_reward'] = float(best_reward)  # 确保是 Python 的 float 类型
            data['best_mse'] = float(best_mse)
            with open('model_config.json', 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            print("best_thicknesses best_reward 已成功更新。")
        else:
            print("best_thicknesses best_reward 未更新。")
        # 添加反射率对比图
        print("\n绘制反射率对比图...")
        plot_reflection_comparison(env, best_thicknesses, env.target_thicknesses)
        
        # 在评估后检查是否有相邻相同材料可以合并
        if best_thicknesses is not None:
            material_indices = env.fixed_materials
            materials = [list(env.available_materials.keys())[idx] for idx in material_indices]
            merged_materials, merged_thicknesses = env._merge_adjacent_same_materials(materials, best_thicknesses)
            
            if len(merged_materials) != len(materials):
                print("\n检测到相邻相同材料，合并后的配置:")
                print(f"合并前层数: {len(materials)}, 合并后层数: {len(merged_materials)}")
                print("合并后的材料序列:", merged_materials)
                print("合并后的厚度配置 (nm):")
                for i, (mat, t) in enumerate(zip(merged_materials, merged_thicknesses)):
                    print(f"Layer {i+1} ({mat}): {t*1e9} nm")
                
                # 更新最佳厚度和材料
                best_materials = merged_materials
                best_thicknesses = merged_thicknesses
                
                # 创建新环境用于绘图
                plot_env = ThicknessOptimizationEnv(
                    target_reflection=target_reflection,
                    materials_sequence=best_materials,
                    num_layers=len(best_materials)
                )
                
                # 使用合并后的配置绘图
                plot_reflection_comparison(plot_env, best_thicknesses, plot_env.target_thicknesses)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 