#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：自定义目标反射率
展示如何使用自定义目标反射率优化多层薄膜
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multilayer_film_optimizer.core.environment import ThicknessOptimizationEnv
from multilayer_film_optimizer.core.optimizer import FilmOptimizer
from multilayer_film_optimizer.core.materials import MaterialLibrary
from multilayer_film_optimizer.utils.visualization import Visualizer

def main():
    """主函数"""
    # 设置输出目录
    output_dir = "results/custom_target"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义自定义目标反射率
    # 这里我们创建一个在特定波长范围内具有特定反射率的目标
    custom_target = {
        "wavelength_ranges": [
            {"range": [0.38e-6, 0.5e-6], "target": 0.9},  # 蓝光高反射
            {"range": [0.5e-6, 0.6e-6], "target": 0.1},   # 绿光低反射
            {"range": [0.6e-6, 0.7e-6], "target": 0.9},   # 红光高反射
            {"range": [0.7e-6, 14e-6], "target": 0.0}     # 红外低反射
        ]
    }
    
    # 将自定义目标保存到文件
    target_file = os.path.join(output_dir, "custom_target.json")
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(custom_target, f, indent=4, ensure_ascii=False)
    
    # 设置材料序列
    # 这里我们使用更多的层来实现更复杂的光学特性
    materials_sequence = ["SiO2", "W", "Ge", "SiO2", "W", "Ge", "SiO2"]
    
    # 创建环境
    env = ThicknessOptimizationEnv(
        target_reflection=custom_target,
        materials_sequence=materials_sequence,
        num_layers=len(materials_sequence)
    )
    
    # 创建优化器
    optimizer = FilmOptimizer(
        env=env,
        output_dir=output_dir,
        tensorboard_dir=os.path.join(output_dir, "tensorboard_logs")
    )
    
    # 训练模型
    print("开始训练模型...")
    best_model = optimizer.train(
        total_timesteps=15000,  # 更多的训练步数
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    print(f"模型训练完成，最佳模型已保存到 {output_dir}")
    
    # 评估最佳模型
    print("\n开始评估最佳模型...")
    best_mse, best_reward, best_thicknesses = optimizer.evaluate(
        model=best_model,
        num_episodes=10,
        render=True
    )
    
    # 保存结果
    result = env.save_results(output_dir=output_dir)
    
    # 创建报告
    if result is not None:
        report_path = Visualizer.create_report(
            os.path.join(output_dir, "optimization_result.json"),
            output_dir=os.path.join(output_dir, "reports")
        )
        print(f"优化结果报告已保存到 {report_path}")
    
    # 绘制目标反射率曲线
    plt.figure(figsize=(10, 6))
    
    # 创建波长数组
    wavelength = np.linspace(0.38e-6, 14e-6, 1000)
    
    # 创建目标反射率数组
    target_reflection = np.zeros_like(wavelength)
    for wavelength_range in custom_target["wavelength_ranges"]:
        range_start, range_end = wavelength_range["range"]
        target_value = wavelength_range["target"]
        target_reflection[(wavelength >= range_start) & (wavelength <= range_end)] = target_value
    
    # 绘制目标反射率曲线
    plt.plot(wavelength * 1e6, target_reflection, 'b-', label='Target')
    
    # 设置坐标轴
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Reflectivity')
    plt.title('Custom Target Reflectivity')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, "custom_target.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("程序运行完成")

if __name__ == "__main__":
    main() 