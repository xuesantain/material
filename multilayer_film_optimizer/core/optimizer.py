import os
import time
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

class TensorboardCallback(BaseCallback):
    """用于记录训练过程的回调函数"""
    
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
        
        # 确保日志被写入
        self.logger.dump(step=self.num_timesteps)
        return True

class FilmOptimizer:
    """薄膜优化器类，用于训练和评估模型"""
    
    def __init__(self, env, output_dir="results", tensorboard_dir="tensorboard_logs"):
        """
        初始化优化器
        
        参数:
            env: 优化环境
            output_dir: 输出目录
            tensorboard_dir: Tensorboard日志目录
        """
        self.env = env
        self.output_dir = output_dir
        self.tensorboard_dir = tensorboard_dir
        self.model = None
        self.best_model = None
        self.config = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # 检查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"训练设备: {self.device}")
        
        # 检查环境
        check_env(env)
        
    def train(self, total_timesteps=10000, learning_rate=1e-3, n_steps=2048, 
              batch_size=64, n_epochs=10, continue_training=False):
        """
        训练模型
        
        参数:
            total_timesteps: 总训练步数
            learning_rate: 学习率
            n_steps: 每次更新的步数
            batch_size: 批量大小
            n_epochs: 每次更新的训练轮数
            continue_training: 是否继续训练现有模型
            
        返回:
            训练后的最佳模型
        """
        model_path = os.path.join(self.output_dir, "best_model.zip")
        config_path = os.path.join(self.output_dir, "model_config.json")
        
        if continue_training and os.path.exists(model_path):
            print("加载已有模型继续训练...")
            self.model = PPO.load(model_path, env=self.env, device=self.device)
            self.model.learning_rate = learning_rate / 2  # 降低学习率
            
            # 加载配置
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.model = PPO(
                "MlpPolicy", 
                self.env, 
                verbose=1,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=0.99,
                ent_coef=0.05,
                clip_range=0.2,
                vf_coef=0.5,
                max_grad_norm=0.5,
                gae_lambda=0.95,
                normalize_advantage=True,
                tensorboard_log=self.tensorboard_dir,
                device=self.device
            )
            
            # 创建初始配置
            self.config = {
                "target_specs": {
                    "wavelength_ranges": self.env.target_reflection["wavelength_ranges"]
                },
                "materials": self.env.materials_sequence,
                "fixed_order": [int(x) for x in self.env.fixed_materials],
                "material_paths": self.env.material_paths,
                "best_mse": float('inf'),
                "best_thicknesses": None,
                "best_reward": float('-inf'),
                "wavelength_range": {
                    "min": float(self.env.wl.min()),
                    "max": float(self.env.wl.max()),
                    "points": int(len(self.env.wl))
                }
            }
        
        # 设置迭代次数
        n_iterations = 10
        timesteps_per_iteration = total_timesteps // n_iterations
        
        # 创建callback用于记录训练过程
        callback = TensorboardCallback()
        
        # 训练循环
        for i in range(n_iterations):
            print(f"\n训练迭代 {i+1}/{n_iterations}")
            
            self.model.learn(
                total_timesteps=timesteps_per_iteration,
                reset_num_timesteps=False,
                callback=callback,
                tb_log_name=f"PPO_run_{i+1}",
                progress_bar=True
            )
            
            # 评估当前模型
            current_mse, current_reward, current_thicknesses = self.evaluate(num_episodes=5, render=False)
            
            # 如果当前模型更好，则保存为最佳模型
            if current_mse < self.config["best_mse"]:
                self.config["best_mse"] = float(current_mse)
                self.config["best_thicknesses"] = [float(t) for t in current_thicknesses]
                self.config["best_reward"] = float(current_reward)
                
                # 保存模型和配置
                self.model.save(os.path.join(self.output_dir, "best_model"))
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
                    
                self.best_model = PPO.load(os.path.join(self.output_dir, "best_model"), env=self.env)
                print(f"最佳模型已更新，MSE: {current_mse}")
        
        # 保存最终模型
        self.model.save(os.path.join(self.output_dir, "final_model"))
        
        # 如果没有最佳模型，使用最终模型
        if self.best_model is None:
            self.best_model = self.model
            
        return self.best_model
    
    def evaluate(self, model=None, num_episodes=10, render=True):
        """
        评估模型性能
        
        参数:
            model: 要评估的模型，如果为None则使用当前最佳模型
            num_episodes: 评估的episode数量
            render: 是否渲染结果
            
        返回:
            最佳MSE、最佳奖励、最佳厚度
        """
        if model is None:
            if self.best_model is not None:
                model = self.best_model
            elif self.model is not None:
                model = self.model
            else:
                raise ValueError("没有可用的模型进行评估")
        
        mean_reward = 0
        best_reward = float('-inf')
        best_thicknesses = None
        best_mse = float('inf')
        best_episode = -1
        
        # 获取材料名称列表
        material_names = list(self.env.available_materials.keys())
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            step_count = 0
            episode_best_reward = float('-inf')
            episode_best_thicknesses = None
            episode_best_mse = float('inf')
            
            while not done and step_count < 100:
                action, _ = model.predict(obs)
                obs, reward, done, _, _ = self.env.step(action)
                current_mse = np.mean((self.env.current_reflection - self.env.target['target'][0])**2)
                
                if current_mse < episode_best_mse:
                    episode_best_reward = reward
                    episode_best_thicknesses = self.env.current_thicknesses.copy()
                    episode_best_mse = current_mse
                
                step_count += 1
            
            if episode_best_mse < best_mse:
                best_reward = episode_best_reward
                best_thicknesses = episode_best_thicknesses.copy()
                best_mse = episode_best_mse
                best_episode = episode + 1
                
            if render:
                self.env.render()
                time.sleep(0.1)
                
            mean_reward += episode_best_reward
            
            print(f"\nEpisode {episode + 1}: 最佳奖励 = {episode_best_reward}")
            print(f"MSE = {episode_best_mse}")
            print("本次episode最佳厚度 (nm):")
            for i, t in enumerate(episode_best_thicknesses):
                material_name = material_names[self.env.fixed_materials[i]]
                print(f"Layer {i+1} ({material_name}): {t*1e9:.2f} nm")
        
        print(f"\n在所有{num_episodes}个episodes中:")
        print(f"最佳结果出现在Episode {best_episode}")
        print(f"最佳奖励值: {best_reward}")
        print(f"最佳MSE: {best_mse}")
        print("最佳厚度配置:")
        for i, t in enumerate(best_thicknesses):
            material_name = material_names[self.env.fixed_materials[i]]
            print(f"Layer {i+1} ({material_name}): {t*1e9:.2f} nm")
        
        # 保存评估结果
        eval_dir = os.path.join(self.output_dir, "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        eval_file_path = os.path.join(eval_dir, f"evaluation_{timestamp}.txt")
        
        with open(eval_file_path, 'w', encoding="utf-8") as f:
            f.write(f"评估结果:\n")
            f.write(f"评估episode数量: {num_episodes}\n")
            f.write(f"最佳结果出现在Episode {best_episode}\n")
            f.write(f"最佳奖励值: {best_reward}\n")
            f.write(f"最佳MSE: {best_mse}\n")
            f.write("\n最佳厚度配置:\n")
            for i, t in enumerate(best_thicknesses):
                material_name = material_names[self.env.fixed_materials[i]]
                f.write(f"Layer {i+1} ({material_name}): {t*1e9:.2f} nm\n")
        
        # 绘制并保存反射率对比图
        self._plot_reflection_comparison(best_thicknesses, eval_dir, timestamp)
        
        mean_reward /= num_episodes
        return best_mse, best_reward, best_thicknesses
    
    def _plot_reflection_comparison(self, thicknesses, output_dir, timestamp):
        """
        绘制反射率对比图
        
        参数:
            thicknesses: 厚度配置
            output_dir: 输出目录
            timestamp: 时间戳
        """
        # 计算目标反射率
        target_reflection = self.env.target_array[0]
        
        # 计算最佳配置的反射率
        self.env.env.reset()
        for material_idx, thickness in zip(self.env.fixed_materials, thicknesses):
            action = self.env.env.create_action(material_idx + 1, thickness, is_normalized=False)
            self.env.env.step(action)
        best_reflection = self.env.env.simulation[0]
        
        # 绘制反射率对比
        plt.figure(figsize=(10, 6))
        plt.plot(self.env.wl * 1e6, target_reflection, 'b-', label='Target')
        plt.plot(self.env.wl * 1e6, best_reflection, 'r--', label='Optimized')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectivity')
        plt.title('Reflectivity Comparison')
        plt.legend()
        plt.grid(True)
        
        # 保存厚度信息到文本文件
        thickness_file_path = os.path.join(output_dir, f"thickness_info_{timestamp}.txt")
        
        with open(thickness_file_path, 'w', encoding="utf-8") as f:
            f.write("最优厚度配置:\n")
            material_names = list(self.env.available_materials.keys())
            for i, t in enumerate(thicknesses):
                material_name = material_names[self.env.fixed_materials[i]]
                f.write(f"Layer {i+1} ({material_name}): {t*1e9:.2f} nm\n")
        
        # 保存反射率图
        plt.savefig(os.path.join(output_dir, f"reflection_comparison_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_model(self, model_path="best_model", config_path="model_config.json"):
        """
        加载已训练的模型和配置
        
        参数:
            model_path: 模型路径
            config_path: 配置路径
            
        返回:
            加载的模型和配置
        """
        try:
            # 检查路径是否为相对路径
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.output_dir, model_path)
            
            if not os.path.isabs(config_path):
                config_path = os.path.join(self.output_dir, config_path)
            
            # 加载模型并指定设备
            model = PPO.load(model_path, env=self.env, device=self.device)
            
            # 加载配置
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            self.model = model
            self.best_model = model
            self.config = config
            
            return model, config
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None, None
    
    def is_similar_target(self, target1, target2, threshold=0.2):
        """
        比较两个目标是否相似
        
        参数:
            target1: 第一个目标
            target2: 第二个目标
            threshold: 相似度阈值
            
        返回:
            是否相似
        """
        try:
            # 比较波长范围数量
            ranges1 = target1["wavelength_ranges"]
            ranges2 = target2["target_specs"]["wavelength_ranges"]
            
            if len(ranges1) != len(ranges2):
                print("波长范围数量不同")
                return False
                
            # 比较每个波长范围
            for range1, range2 in zip(ranges1, ranges2):
                start1, end1 = range1["range"]
                start2, end2 = range2["range"]
                
                start_diff = abs(start1 - start2)
                end_diff = abs(end1 - end2)
                
                if start_diff > threshold * start1 or end_diff > threshold * end1:
                    print(f"波长范围差异过大: {range1['range']} vs {range2['range']}")
                    return False
                
                # 比较目标反射率
                target_diff = abs(range1["target"] - range2["target"])
                if target_diff > threshold:
                    print(f"反射率目标差异过大: {range1['target']} vs {range2['target']}")
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
