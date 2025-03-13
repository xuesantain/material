import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from modules.env import ThicknessOptimizationEnv, plot_reflection_comparison
from modules.training import load_model_and_config, is_similar_target

def main():
    # 定义目标反射率
    target_reflection = {
        "high_reflection": {
            "wavelength": [
                [0.38e-6, 0.8e-6],
                [5e-6, 8e-6]
            ],
            "target": 5.0
        },
        "materials": ["W", "Ge", "SiO2"]
    }
    
    # 加载模型和配置
    model, config = load_model_and_config()
    
    if model is None or config is None:
        print("无法加载模型或配置，请先训练模型")
        return
    
    # 检查目标是否相似
    if not is_similar_target(target_reflection, config):
        print("目标反射率与训练模型时使用的目标不同，建议重新训练模型")
        return
    
    # 创建环境
    materials_sequence = [config["materials"][idx] for idx in config["fixed_order"]]
    
    # 创建环境
    env = ThicknessOptimizationEnv(
        target_reflection={
            "wavelength_ranges": [
                {"range": [0.38e-6, 0.8e-6], "target": 0.0},
                {"range": [3e-6, 5e-6], "target": 1.0},
                {"range": [5e-6, 8e-6], "target": 0.0},
                {"range": [8e-6, 14e-6], "target": 1.0}
            ]
        },
        materials_sequence=materials_sequence
    )
    
    # 使用模型预测最佳厚度
    obs, _ = env.reset()
    best_reward = float('-inf')
    best_thicknesses = None
    best_mse = float('inf')
    
    # 运行多次预测，选择最佳结果
    for _ in range(10):
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            current_mse = np.mean((env.current_reflection - env.target['target'][0])**2)
            
            if current_mse < best_mse:
                best_reward = reward
                best_thicknesses = env.current_thicknesses.copy()
                best_mse = current_mse
            
            step_count += 1
    
    # 打印最佳结果
    print(f"最佳MSE: {best_mse}")
    print(f"最佳奖励: {best_reward}")
    print("最佳厚度配置 (nm):")
    material_names = [config["materials"][idx] for idx in config["fixed_order"]]
    for i, (mat, t) in enumerate(zip(material_names, best_thicknesses)):
        print(f"Layer {i+1} ({mat}): {t*1e9:.2f} nm")
    
    # 检查是否有相邻相同材料可以合并
    materials = [config["materials"][idx] for idx in config["fixed_order"]]
    merged_materials, merged_thicknesses = env._merge_adjacent_same_materials(materials, best_thicknesses)
    
    if len(merged_materials) != len(materials):
        print("\n检测到相邻相同材料，合并后的配置:")
        print(f"合并前层数: {len(materials)}, 合并后层数: {len(merged_materials)}")
        print("合并后的材料序列:", merged_materials)
        print("合并后的厚度配置 (nm):")
        for i, (mat, t) in enumerate(zip(merged_materials, merged_thicknesses)):
            print(f"Layer {i+1} ({mat}): {t*1e9:.2f} nm")
        
        # 更新最佳厚度和材料
        best_materials = merged_materials
        best_thicknesses = merged_thicknesses
        
        # 创建新环境用于绘图
        plot_env = ThicknessOptimizationEnv(
            target_reflection={
                "wavelength_ranges": [
                    {"range": [0.38e-6, 0.8e-6], "target": 0.0},
                    {"range": [3e-6, 5e-6], "target": 1.0},
                    {"range": [5e-6, 8e-6], "target": 0.0},
                    {"range": [8e-6, 14e-6], "target": 1.0}
                ]
            },
            materials_sequence=best_materials,
            num_layers=len(best_materials)
        )
        
        # 使用合并后的配置绘图
        plot_reflection_comparison(plot_env, best_thicknesses, plot_env.target_thicknesses)
    else:
        # 绘制反射率对比图
        plot_reflection_comparison(env, best_thicknesses, env.target_thicknesses)

if __name__ == "__main__":
    main() 