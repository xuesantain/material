#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：优化红外滤波器
展示如何使用多层薄膜优化器优化红外滤波器
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
from multilayer_film_optimizer.config.default_config import PRESET_TARGETS

def main():
    """主函数"""
    # 设置输出目录
    output_dir = "results/ir_filter"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用预设的红外带通滤波器目标
    target_reflection = PRESET_TARGETS["ir_bandpass"]["target_reflection"]
    
    # 设置材料序列
    materials_sequence = ["W", "Ge", "SiO2", "W", "Ge"]
    
    # 创建环境
    env = ThicknessOptimizationEnv(
        target_reflection=target_reflection,
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
        total_timesteps=10000,
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
    
    print("程序运行完成")

if __name__ == "__main__":
    main() 