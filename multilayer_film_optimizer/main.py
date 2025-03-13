#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多层薄膜优化器主程序
用于优化多层薄膜的厚度，以达到特定的光学反射率目标
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilayer_film_optimizer.core.environment import ThicknessOptimizationEnv
from multilayer_film_optimizer.core.optimizer import FilmOptimizer
from multilayer_film_optimizer.core.materials import MaterialLibrary
from multilayer_film_optimizer.utils.visualization import Visualizer
from multilayer_film_optimizer.config.default_config import (
    DEFAULT_TARGET_REFLECTION,
    DEFAULT_WAVELENGTH_RANGE,
    DEFAULT_THICKNESS_RANGE,
    DEFAULT_MATERIALS_SEQUENCE,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    PRESET_TARGETS
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多层薄膜优化器")
    
    # 基本参数
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="输出目录")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "optimize"],
                        help="运行模式: train(训练), evaluate(评估), optimize(优化)")
    
    # 目标反射率参数
    parser.add_argument("--target_preset", type=str, default="custom",
                        choices=list(PRESET_TARGETS.keys()),
                        help="预设目标反射率")
    parser.add_argument("--target_file", type=str, default=None,
                        help="目标反射率配置文件路径")
    
    # 材料参数
    parser.add_argument("--materials", type=str, nargs="+", default=None,
                        help="材料序列")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="薄膜层数")
    
    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_TRAINING_CONFIG["total_timesteps"],
                        help="总训练步数")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_TRAINING_CONFIG["learning_rate"],
                        help="学习率")
    parser.add_argument("--continue_training", action="store_true",
                        help="是否继续训练现有模型")
    
    # 评估参数
    parser.add_argument("--num_episodes", type=int, default=DEFAULT_EVALUATION_CONFIG["num_episodes"],
                        help="评估的episode数量")
    parser.add_argument("--render", action="store_true",
                        help="是否渲染结果")
    parser.add_argument("--model_path", type=str, default="best_model",
                        help="模型路径")
    
    return parser.parse_args()

def load_target_reflection(args):
    """加载目标反射率配置"""
    if args.target_file is not None and os.path.exists(args.target_file):
        # 从文件加载
        with open(args.target_file, 'r', encoding='utf-8') as f:
            target_reflection = json.load(f)
    else:
        # 使用预设
        target_reflection = PRESET_TARGETS[args.target_preset]["target_reflection"]
    
    return target_reflection

def train(args):
    """训练模型"""
    print("开始训练模型...")
    
    # 加载目标反射率
    target_reflection = load_target_reflection(args)
    
    # 设置材料序列
    materials_sequence = args.materials if args.materials is not None else DEFAULT_MATERIALS_SEQUENCE
    
    # 创建环境
    env = ThicknessOptimizationEnv(
        target_reflection=target_reflection,
        materials_sequence=materials_sequence,
        num_layers=args.num_layers,
        min_thickness=DEFAULT_THICKNESS_RANGE["min"],
        max_thickness=DEFAULT_THICKNESS_RANGE["max"],
        wavelength_range=DEFAULT_WAVELENGTH_RANGE
    )
    
    # 创建优化器
    optimizer = FilmOptimizer(
        env=env,
        output_dir=args.output_dir,
        tensorboard_dir=os.path.join(args.output_dir, "tensorboard_logs")
    )
    
    # 训练模型
    best_model = optimizer.train(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        continue_training=args.continue_training
    )
    
    print(f"模型训练完成，最佳模型已保存到 {args.output_dir}")
    
    # 评估最佳模型
    print("\n开始评估最佳模型...")
    best_mse, best_reward, best_thicknesses = optimizer.evaluate(
        model=best_model,
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # 保存结果
    result = env.save_results(output_dir=args.output_dir)
    
    # 创建报告
    if result is not None:
        report_path = Visualizer.create_report(
            os.path.join(args.output_dir, "optimization_result.json"),
            output_dir=os.path.join(args.output_dir, "reports")
        )
        print(f"优化结果报告已保存到 {report_path}")
    
    return best_model, env

def evaluate(args):
    """评估模型"""
    print("开始评估模型...")
    
    # 加载目标反射率
    target_reflection = load_target_reflection(args)
    
    # 设置材料序列
    materials_sequence = args.materials if args.materials is not None else DEFAULT_MATERIALS_SEQUENCE
    
    # 创建环境
    env = ThicknessOptimizationEnv(
        target_reflection=target_reflection,
        materials_sequence=materials_sequence,
        num_layers=args.num_layers,
        min_thickness=DEFAULT_THICKNESS_RANGE["min"],
        max_thickness=DEFAULT_THICKNESS_RANGE["max"],
        wavelength_range=DEFAULT_WAVELENGTH_RANGE
    )
    
    # 创建优化器
    optimizer = FilmOptimizer(
        env=env,
        output_dir=args.output_dir
    )
    
    # 加载模型
    model_path = os.path.join(args.output_dir, args.model_path)
    config_path = os.path.join(args.output_dir, "model_config.json")
    
    model, config = optimizer.load_model(model_path, config_path)
    
    if model is None:
        print(f"无法加载模型 {model_path}")
        return None, env
    
    # 评估模型
    best_mse, best_reward, best_thicknesses = optimizer.evaluate(
        model=model,
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # 保存结果
    result = env.save_results(output_dir=args.output_dir)
    
    # 创建报告
    if result is not None:
        report_path = Visualizer.create_report(
            os.path.join(args.output_dir, "optimization_result.json"),
            output_dir=os.path.join(args.output_dir, "reports")
        )
        print(f"优化结果报告已保存到 {report_path}")
    
    return model, env

def optimize(args):
    """优化薄膜厚度"""
    print("开始优化薄膜厚度...")
    
    # 加载目标反射率
    target_reflection = load_target_reflection(args)
    
    # 设置材料序列
    materials_sequence = args.materials if args.materials is not None else DEFAULT_MATERIALS_SEQUENCE
    
    # 创建环境
    env = ThicknessOptimizationEnv(
        target_reflection=target_reflection,
        materials_sequence=materials_sequence,
        num_layers=args.num_layers,
        min_thickness=DEFAULT_THICKNESS_RANGE["min"],
        max_thickness=DEFAULT_THICKNESS_RANGE["max"],
        wavelength_range=DEFAULT_WAVELENGTH_RANGE
    )
    
    # 检查是否有现有模型
    model_path = os.path.join(args.output_dir, args.model_path)
    config_path = os.path.join(args.output_dir, "model_config.json")
    
    if os.path.exists(model_path + ".zip") and os.path.exists(config_path):
        # 加载现有模型
        print(f"加载现有模型 {model_path}...")
        optimizer = FilmOptimizer(
            env=env,
            output_dir=args.output_dir
        )
        model, config = optimizer.load_model(model_path, config_path)
        
        if model is not None:
            # 检查目标是否相似
            if optimizer.is_similar_target(target_reflection, config):
                print("目标相似，使用现有模型进行评估...")
                best_mse, best_reward, best_thicknesses = optimizer.evaluate(
                    model=model,
                    num_episodes=args.num_episodes,
                    render=args.render
                )
                
                # 保存结果
                result = env.save_results(output_dir=args.output_dir)
                
                # 创建报告
                if result is not None:
                    report_path = Visualizer.create_report(
                        os.path.join(args.output_dir, "optimization_result.json"),
                        output_dir=os.path.join(args.output_dir, "reports")
                    )
                    print(f"优化结果报告已保存到 {report_path}")
                
                return model, env
    
    # 没有现有模型或目标不相似，训练新模型
    print("训练新模型...")
    return train(args)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式运行
    if args.mode == "train":
        model, env = train(args)
    elif args.mode == "evaluate":
        model, env = evaluate(args)
    elif args.mode == "optimize":
        model, env = optimize(args)
    else:
        print(f"未知模式: {args.mode}")
        return
    
    print("程序运行完成")

if __name__ == "__main__":
    main()
