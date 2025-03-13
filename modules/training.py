import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self):
        # 获取环境实例
        env = self.training_env.envs[0].unwrapped
        
        # 计算实际的匹配度
        mse = np.mean((env.current_reflection - env.target['target'][0])**2)
        similarity = 1.0 / (1.0 + mse)  # 转换为0-1之间的相似度
        
        # 记录反射率匹配度
        self.logger.record("custom/mse", mse)
        self.logger.record("custom/similarity", similarity)
        
        # 记录当前厚度
        for i, thickness in enumerate(env.current_thicknesses):
            self.logger.record(f"custom/thickness_layer_{i+1}", thickness * 1e9)
            # 记录与目标厚度的差异
            # diff = (thickness - env.target_thicknesses[i]) * 1e9
            # self.logger.record(f"custom/thickness_diff_{i+1}", diff)
        
        # 确保日志被写入
        self.logger.dump(step=self.num_timesteps)
        return True

def train_model(env, continue_training=False):
    # 检查CUDA是否可用并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # 检查环境
    check_env(env)
    
    if continue_training and os.path.exists("best_model.zip"):
        print("加载已有模型继续训练...")
        model = PPO.load("best_model.zip", env=env, device=device)  # 指定设备
        model.learning_rate = 1e-4
    else:
        model = PPO("MlpPolicy", 
                   env, 
                   verbose=1,
                   learning_rate=1e-3,     # 更大的学习率
                   n_steps=8192,          # 更多的步数
                   batch_size=1024,        # 更大的批量
                   n_epochs=30,            
                   gamma=0.99,             
                   ent_coef=0.05,          # 更大的熵系数
                   clip_range=0.3,         # 更大的裁剪范围
                   vf_coef=0.5,
                   max_grad_norm=0.5,
                   gae_lambda=0.9,         # 较小的GAE-Lambda
                   normalize_advantage=True,
                   tensorboard_log="./thickness_optimization_tensorboard/",
                   device=device)  # 指定设备

    # 设置总步数
    total_timesteps = 7280  # 增加训练步数以允许更多探索
    n_iterations = 30
    timesteps_per_iteration = total_timesteps // n_iterations
    
    # 跟踪已执行的总步数
    total_steps_taken = 0
    
    # 创建callback用于记录训练过程
    callback = TensorboardCallback()
    
    # 添加用于跟踪最佳模型的变量
    best_model = None
    
    # 添加配置初始化
    if not continue_training:
        config = {
            "target_specs": {
                "high_reflection": {
                # 将NumPy数组转换为Python列表
                "wavelength": [[float(x[0]), float(x[1])] for x in env.high_reflection_ranges],
                "target": float(env.high_reflection_target)
                }
            },
            "materials": ["W", "Ge", "SiO2"],  # 材料类型
            "fixed_order": [int(x) for x in env.fixed_materials],    # 转换为Python列表
            "material_paths": env.material_paths,
            "best_mse": float('inf'),
            "best_thicknesses": None,
            "best_reward": float('-inf'),
            "wavelength_range": {
                "min": float(env.wl.min()),
                "max": float(env.wl.max()),
                "points": int(len(env.wl))
            }
        }
    
    for i in range(n_iterations):
        print(f"\nTraining iteration {i+1}/{n_iterations}")
        # 计算这次迭代应该执行的步数
        remaining_steps = total_timesteps - total_steps_taken
        if remaining_steps <= 0:
            break
            
        steps_this_iteration = min(timesteps_per_iteration, remaining_steps)
        
        model.learn(
            total_timesteps=steps_this_iteration,
            reset_num_timesteps=True,
            callback=callback,
            tb_log_name=f"PPO_run_{i+1}",
            progress_bar=True,           # 显示进度条
            log_interval=1              # 日志记录间隔
        )
        
        total_steps_taken += steps_this_iteration
        
        # 评估当前模型
        current_best_mse, current_reward, current_thicknesses = evaluate_model(model, env, num_episodes=10, render=False)
        
        
        if continue_training:
            print(f"current_mse: {current_best_mse}")
            with open("model_config.json", "r") as f:
                config = json.load(f)
            if config['best_mse'] > current_best_mse:  # 使用current_mse
                config['best_mse'] = float(current_best_mse)
                config['best_thicknesses'] = [float(x) for x in current_thicknesses]  # 转换为列表
                config['best_reward'] = float(current_reward)
                with open("model_config.json", "w") as f:
                    json.dump(config, f, indent=4)
                model.save("best_model")
                best_model = PPO.load("best_model", env=env)
                print(f"best_model 更新成功")
        # 如果当前模型更好，则保存为最佳模型
        if current_best_mse < config["best_mse"]:
            best_mse = current_best_mse
            model.save("best_model")
            best_model = PPO.load("best_model", env=env)
            # 确保所有数值都转换为Python原生类型
            config["best_mse"] = float(best_mse)
            config["best_thicknesses"] = [float(x) for x in current_thicknesses]  # 转换为列表
            config["best_reward"] = float(current_reward)
            with open("model_config.json", "w") as f:
                json.dump(config, f, indent=4)
    
    # 保存最终模型
    model.save("final_model")
    if best_model is None:
        best_model = model
        print(f"best_model 没有更新")
    return best_model, env, config

def evaluate_model(model, env, num_episodes=10, render=True):
    """评估模型性能"""
    mean_reward = 0
    best_reward = float('-inf')
    best_thicknesses = None
    best_mse = float('inf')
    best_episode = -1
    
    # 获取材料名称列表
    material_names = list(env.available_materials.keys())
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_best_reward = float('-inf')
        episode_best_thicknesses = None
        episode_best_mse = float('inf')
        
        while not done and step_count < 100:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            current_mse = np.mean((env.current_reflection - env.target['target'][0])**2)
            
            if current_mse < episode_best_mse:
                episode_best_reward = reward
                episode_best_thicknesses = env.current_thicknesses.copy()
                episode_best_mse = current_mse
            
            step_count += 1
        
        if episode_best_mse < best_mse:
            best_reward = episode_best_reward
            best_thicknesses = episode_best_thicknesses.copy()
            best_mse = episode_best_mse
            best_episode = episode + 1
            
        if render:
            env.render()
            time.sleep(0.1)
            
        mean_reward += episode_best_reward
        
        print(f"\nEpisode {episode + 1}: Best Reward = {episode_best_reward}")
        print(f"mse = {episode_best_mse}")
        print("Best thicknesses in this episode (nm):")
        for i, t in enumerate(episode_best_thicknesses):
            print(f"Layer {i+1} ({material_names[env.fixed_materials[i]]}):"
                     f" {t*1e9} nm")
    
    print(f"\n在所有{num_episodes}个episodes中:")
    print(f"最佳结果出现在Episode {best_episode}")
    print(f"最佳奖励值: {best_reward}")
    print(f"最佳MSE: {best_mse}")
    print("最佳厚度配置:")
    for i, t in enumerate(best_thicknesses):
        print(f"Layer {i+1} ({material_names[env.fixed_materials[i]]}):"
                 f" {t*1e9} nm")
    
    os.makedirs('./runingresult', exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    eval_file_path = f'./runingresult/evaluation_{timestamp}.txt'
    
    with open(eval_file_path, 'w') as f:
        f.write(f"评估结果:\n")
        f.write(f"评估episode数量: {num_episodes}\n")
        f.write(f"最佳结果出现在Episode {best_episode}\n")
        f.write(f"最佳奖励值: {best_reward}\n")
        f.write(f"最佳MSE: {best_mse}\n")
        f.write("\n最佳厚度配置:\n")
        for i, t in enumerate(best_thicknesses):
            material_name = material_names[env.fixed_materials[i]]
            f.write(f"Layer {i+1} ({material_name}): {t*1e9:.2f} nm\n")
    
    mean_reward /= num_episodes
    return best_mse, best_reward, best_thicknesses

def load_model_and_config(model_path="best_model", config_path="model_config.json"):
    """加载已训练的模型和配置"""
    try:
        # 检查CUDA是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型并指定设备
        model = PPO.load(model_path, device=device)
        
        # 加载配置
        with open(config_path, "r") as f:
            config = json.load(f)
        
        return model, config
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None

def is_similar_target(target1, target2, threshold=0.2):
    """比较两个目标是否相似"""
    try:
        # 比较反射率目标
        hr1_ranges = target1["high_reflection"]["wavelength"]
        hr2_ranges = target2["target_specs"]["high_reflection"]["wavelength"]
        
        # 比较波长范围
        if len(hr1_ranges) != len(hr2_ranges):
            print("高反射区域数量不同")
            return False
            
        # 比较每个波长范围
        for range1, range2 in zip(hr1_ranges, hr2_ranges):
            start_diff = abs(range1[0] - range2[0])
            end_diff = abs(range1[1] - range2[1])
            if start_diff > threshold * range1[0] or end_diff > threshold * range1[1]:
                print(f"波长范围差异过大: {range1} vs {range2}")
                return False
        
        # 比较目标反射率
        target_diff = abs(target1["high_reflection"]["target"] - 
                        target2["target_specs"]["high_reflection"]["target"])
        if target_diff > threshold:
            print(f"反射率目标差异过大: {target_diff}")
            return False
            
        # 比较材料类型
        if target1["materials"] != target2["materials"]:
            print("材料类型不同，需要重新训练模型")
            return False
            
        # 比较材料堆叠顺序
        if target1["fixed_order"] != target2["fixed_order"]:
            print("材料堆叠顺序不同，需要重新训练模型")
            return False
            
        return True
        
    except Exception as e:
        print(f"比较目标时出错: {str(e)}")
        return False 