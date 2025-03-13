import numpy as np
from pathlib import Path
import gymnasium
import tmm_fast.gym_multilayerthinfilm as mltf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
import json
import torch  # 添加到文件开头的导入部分
from copy import deepcopy
from materials_config import AVAILABLE_MATERIALS
import random

class ThicknessOptimizationEnv(gymnasium.Env):
    def __init__(self, target_reflection, materials_sequence=None, num_layers=8):
        """
        初始化环境
        target_reflection: 目标反射率曲线
        materials_sequence: 材料序列，如果为None则自动选择
        num_layers: 薄膜层数
        """
        self.num_layers = num_layers
        self.target_reflection = target_reflection
        
        # 设置波长范围
        self.wl = np.linspace(380, 14000, 1363) * (10**(-9))
        
        # 加载所有可用材料
        self.available_materials = AVAILABLE_MATERIALS
        self.material_paths = [mat_info["path"] for mat_info in AVAILABLE_MATERIALS.values()]
        
        # 获取所有材料的折射率数据
        self.N = mltf.get_N(self.material_paths, 
                           self.wl.min() * 1e9, 
                           self.wl.max() * 1e9, 
                           points=len(self.wl), 
                           complex_n=True)
        
        if materials_sequence is None:
            # 自动选择材料序列
            self.materials_sequence = self._optimize_material_sequence()
        else:
            # 使用指定的材料序列，并合并相邻相同材料
            self.materials_sequence = materials_sequence
            
        # 更新fixed_materials为材料索引
        self.fixed_materials = [list(AVAILABLE_MATERIALS.keys()).index(mat) for mat in self.materials_sequence]
        
        # 更新实际层数
        self.num_layers = len(self.materials_sequence)
        
        # STACK-plt.py中的目标厚度
        self.target_thicknesses = np.array([85e-9, 5e-9, 71e-9, 500e-9, 100e-9])
        
        # 创建临时环境来计算目标反射率
        temp_env = mltf.MultiLayerThinFilm(
            self.N,
            self.num_layers,
            {'direction': np.array([0]), 'spectrum': self.wl, 
             'target': np.zeros((1, len(self.wl))), 'mode': 'reflectivity'},
            max_thickness=500e-9,
            min_thickness=5e-9,
            sparse_reward=False
        )
        
        # 计算目标反射率
        temp_env.reset()
        for material_idx, thickness in zip(self.fixed_materials, self.target_thicknesses):
            action = temp_env.create_action(material_idx + 1, thickness, is_normalized=False)
            temp_env.step(action)
        target_array = temp_env.simulation.copy()
        
        # 创建目标反射率数组
        target_array = np.zeros_like(temp_env.simulation)  # 初始化为0
        print(target_array.shape)
        
        # 使用传入的target_reflection设置目标反射率
        for wavelength_range in self.target_reflection["wavelength_ranges"]:
            range_start, range_end = wavelength_range["range"]
            target_value = wavelength_range["target"]
            target_array[0, (self.wl >= range_start) & (self.wl <= range_end)] = target_value
        
        # 看曲线是不是符合要求
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.wl * 1e6, target_array[0], label='Reflectivity', color='b')
        # plt.xlabel('Wavelength (μm)')
        # plt.ylabel('Reflectivity')
        # plt.title('Reflectivity vs Wavelength')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


        self.target_array = target_array
        # 定义关注的波长范围--需要修改--------------------
        self.high_reflection_ranges = [
            [0.38e-6, 0.8e-6],  # 第一个高反射区域
            [5e-6, 8e-6]        # 第二个高反射区域
        ]
        self.high_reflection_target = 5.0  # 高反射目标
        
        # 设置权重
        weights = np.ones_like(target_array)
        # 为每个高反射区域设置权重
        for range_start, range_end in self.high_reflection_ranges:
            weights[0, (self.wl >= range_start) & (self.wl <= range_end)] = 6.0
        weights[0, (self.wl >= 0.8) & (self.wl <= 3)] = 0.0
        weights[0, (self.wl >= 8) & (self.wl <= 14)] = 2.5
        # ------------------------------------
        # 创建target字典
        self.target = {
            'direction': np.array([0]),  # 固定入射角为0度
            'spectrum': self.wl,
            'target': target_array,
            'mode': 'reflectivity'
        }
        
        # 创建基础环境--需要修改最大厚度最小厚度
        self.env = mltf.MultiLayerThinFilm(
            self.N, 
            self.num_layers,
            self.target,
            weights=weights,
            max_thickness=500e-9,  # 最大厚度500nm
            min_thickness=5e-9,    # 最小厚度5nm
            sparse_reward=False
        )
        
        # 定义动作空间 (每层的厚度)
        self.action_space = gymnasium.spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_layers,),
            dtype=np.float32
        )
        
        # 定义观察空间
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.wl),),
            dtype=np.float32
        )

    def _optimize_material_sequence(self):
        """使用遗传算法优化材料序列"""
        import random
        from copy import deepcopy
        
        population_size = 20
        generations = 30
        materials = list(self.available_materials.keys())
        
        # 生成初始种群
        population = []
        for _ in range(population_size):
            sequence = [random.choice(materials) for _ in range(self.num_layers)]
            population.append(sequence)
        
        def evaluate_sequence(sequence):
            # 临时设置材料序列
            temp_materials = [list(self.available_materials.keys()).index(mat) for mat in sequence]
            
            # 创建临时环境评估该序列
            temp_env = deepcopy(self.env)
            temp_env.reset()
            
            # 使用随机厚度进行多次评估
            trials = 5
            total_score = 0
            for _ in range(trials):
                thicknesses = self.env.min_thickness + \
                    (self.env.max_thickness - self.env.min_thickness) * \
                    np.random.random(self.num_layers)
                    
                for material_idx, thickness in zip(temp_materials, thicknesses):
                    action = temp_env.create_action(material_idx + 1, thickness, is_normalized=False)
                    temp_env.step(action)
                
                # 计算得分（考虑多个波长范围的目标）
                reflection = temp_env.simulation[0]
                score = 0
                
                # 评估各个波长范围的性能
                for range_start, range_end in self.high_reflection_ranges:
                    mask = (self.wl >= range_start) & (self.wl <= range_end)
                    target = self.target['target'][0, mask]
                    achieved = reflection[mask]
                    score -= np.mean((achieved - target) ** 2)
                
                total_score += score
                temp_env.reset()
                
            return total_score / trials
        
        best_sequence = None
        best_score = float('-inf')
        
        # 遗传算法主循环
        for generation in range(generations):
            # 评估当前种群
            scores = [evaluate_sequence(seq) for seq in population]
            
            # 更新最佳序列
            max_score_idx = np.argmax(scores)
            if scores[max_score_idx] > best_score:
                best_score = scores[max_score_idx]
                # 合并相邻相同材料
                sequence, _ = self._merge_adjacent_same_materials(population[max_score_idx])
                best_sequence = sequence
            
            # 选择
            selected = []
            for _ in range(population_size // 2):
                tournament = random.sample(list(enumerate(scores)), 3)
                winner_idx = max(tournament, key=lambda x: x[1])[0]
                selected.append(population[winner_idx])
            
            # 交叉
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                crossover_point = random.randint(1, self.num_layers - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                new_population.extend([child1, child2])
            
            # 变异
            for sequence in new_population:
                if random.random() < 0.1:  # 变异概率
                    mutation_point = random.randint(0, self.num_layers - 1)
                    sequence[mutation_point] = random.choice(materials)
            
            population = new_population
            
            print(f"Generation {generation + 1}, Best score: {best_score}")
        
        return best_sequence

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        # 随机初始化厚度
        self.current_thicknesses = self.env.min_thickness + \
            (self.env.max_thickness - self.env.min_thickness) * \
            self.np_random.random(self.num_layers)
        
        # 计算初始反射率并确保类型为float32
        self.current_reflection = self._get_reflection()
        
        # 将observation转换为float32类型
        return self.current_reflection.astype(np.float32), {}

    def step(self, action):
        """执行一步优化"""
        # 将[-1, 1]范围的动作转换为[0, 1]范围
        normalized_action = (action + 1) / 2
        
        # 将归一化的动作转换为实际厚度
        thicknesses = self.env.min_thickness + \
            (self.env.max_thickness - self.env.min_thickness) * normalized_action
        
        self.current_thicknesses = thicknesses
        self.current_reflection = self._get_reflection()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 添加额外的奖励项来鼓励合理的厚度
        thickness_penalty = -0.1 * np.mean(np.abs(normalized_action - 0.5))
        reward += thickness_penalty
        
        # 始终返回False，因为我们希望持续优化
        done = False
        
        # 确保返回的observation是float32类型
        return self.current_reflection.astype(np.float32), reward, done, False, {}

    def _get_reflection(self):
        """获取当前配置的反射率"""
        # 重置环境
        self.env.reset()
        
        # 按固定顺序添加层
        for material_idx, thickness in zip(self.fixed_materials, self.current_thicknesses):
            action = self.env.create_action(material_idx + 1, thickness, is_normalized=False)
            self.env.step(action)
        
        # 确保返回float32类型
        return self.env.simulation[0].astype(np.float32)
# 可以修改
    def _calculate_reward(self):
        """计算奖励，使用软性惩罚"""
        base_reward, _ = self.env.reward_func(
            self.current_reflection.reshape(1, -1),
            self.target['target'],
            self.env.weights,
            self.env.baseline_mse,
            self.env.normalization
        )
        
        # 计算关键区域的匹配度
        critical_mask1 = (self.wl <= 0.8e-6)&(self.wl >= 0.38e-6)
        critical_mask2 = (self.wl >= 5e-6) & (self.wl <= 8e-6)
        critical_mask3 = (self.wl >= 8e-6) & (self.wl <= 14e-6)
        # 使用软性惩罚
        critical_error1 = np.mean(np.abs(
            self.current_reflection[critical_mask1] - 
            self.target['target'][0, critical_mask1]
        ))
        critical_error2 = np.mean(np.abs(
            self.current_reflection[critical_mask2] - 
            self.target['target'][0, critical_mask2]
        ))
        critical_error3 = np.mean(np.abs(
            self.current_reflection[critical_mask3] - 
            self.target['target'][0, critical_mask3]
        ))
        # 只有当误差大于阈值时才添加惩罚
        threshold = 0.05
        if critical_error1 > threshold or critical_error2 > threshold or critical_error3 > 0.1:
            critical_penalty = -0.1 * (critical_error1 + critical_error2) - 0.05*critical_error3
        else:
            critical_penalty = 0
        
        return base_reward + critical_penalty

    def render(self):
        """渲染当前结果"""
        self.env.render()

    def _merge_adjacent_same_materials(self, sequence, thicknesses=None):
        """合并相邻的相同材料层"""
        if len(sequence) <= 1:
            return sequence, thicknesses
            
        merged_sequence = []
        merged_thicknesses = None
        if thicknesses is not None:
            merged_thicknesses = []
            
        i = 0
        while i < len(sequence):
            current_material = sequence[i]
            current_thickness = thicknesses[i] if thicknesses is not None else None
            
            # 查找连续的相同材料
            j = i + 1
            while j < len(sequence) and sequence[j] == current_material:
                if thicknesses is not None:
                    current_thickness += thicknesses[j]
                j += 1
                
            # 添加到合并后的列表
            merged_sequence.append(current_material)
            if merged_thicknesses is not None:
                merged_thicknesses.append(current_thickness)
                
            # 移动到下一个不同的材料
            i = j
            
        return merged_sequence, merged_thicknesses

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

def train_model(menv,continue_training=False):
    # 检查CUDA是否可用并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # 创建环境
    env = menv
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
    # best_reward = float('-inf')
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
        current_best_mse,current_reward, current_thicknesses = evaluate_model(model, env, num_episodes=10, render=False)
        
        
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

def plot_reflection_comparison(env, current_thicknesses, target_thicknesses):
    """绘制最佳反射率与目标反射率的对比图，并保存厚度信息"""
    fig = plt.figure(figsize=(10, 6))
    
    # 计算目标反射率
    env.env.reset()  
    target_reflection = env.target_array[0]  # 取第一维的数据，将(1, 1363)转换为(1363,)
    
    # 计算最佳配置的反射率
    env.env.reset()
    for material_idx, thickness in zip(env.fixed_materials, current_thicknesses):
        action = env.env.create_action(material_idx + 1, thickness, is_normalized=False)
        env.env.step(action)
    best_reflection = env.env.simulation[0]
    
    # 绘制反射率对比
    plt.plot(env.wl * 1e6, target_reflection, 'b-', label='Target')
    plt.plot(env.wl * 1e6, best_reflection, 'r--', label='Optimized')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Reflectivity')
    plt.title('Reflectivity Comparison')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('./runingresult', exist_ok=True)
    # 保存厚度信息到文本文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    thickness_file_path = f'./runingresult/thickness_info_{timestamp}.txt'
    
    with open(thickness_file_path, 'w') as f:
        f.write("最优厚度配置:\n")
        for i, t in enumerate(current_thicknesses):
            material_name = ['W', 'Ge', 'SiO2'][env.fixed_materials[i]]
            f.write(f"Layer {i+1} ({material_name}): {t*1e9:.2f} nm\n")    
    # 保存反射率图
    plt.savefig(f'./runingresult/reflection_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def evaluate_material_combination(self, materials):
    """评估材料组合的性能"""
    score = 0
    
    # 计算材料组合的光学特性
    for wavelength_range in self.target_reflection["wavelength_ranges"]:
        range_start, range_end = wavelength_range["range"]
        target_value = wavelength_range["target"]
        
        # 计算该波长范围内的平均反射率
        mask = (self.wl >= range_start) & (self.wl <= range_end)
        achieved_reflection = self._calculate_reflection(materials, mask)
        
        # 计算与目标的匹配度
        score -= np.abs(achieved_reflection - target_value)
    
    return score

class DynamicMaterialEnv(ThicknessOptimizationEnv):
    def __init__(self, target_reflection, num_layers=5):
        super().__init__(target_reflection, materials_sequence=None, num_layers=num_layers)
        
        # 扩展动作空间，包含材料选择
        self.action_space = gymnasium.spaces.Dict({
            'thicknesses': gymnasium.spaces.Box(
                low=-1, high=1, shape=(self.num_layers,), dtype=np.float32
            ),
            'materials': gymnasium.spaces.MultiDiscrete([
                len(self.available_materials) for _ in range(self.num_layers)
            ])
        })
    
    def step(self, action):
        # 解析动作
        thickness_action = action['thicknesses']
        material_indices = action['materials']
        
        # 更新材料序列
        self.materials_sequence = [
            list(self.available_materials.keys())[idx] 
            for idx in material_indices
        ]
        self.fixed_materials = material_indices
        
        # 执行厚度优化步骤
        normalized_action = (thickness_action + 1) / 2
        thicknesses = self.env.min_thickness + \
            (self.env.max_thickness - self.env.min_thickness) * normalized_action
        
        self.current_thicknesses = thicknesses
        self.current_reflection = self._get_reflection()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 添加材料变化惩罚
        material_change_penalty = -0.05 * np.sum(
            np.array(self.fixed_materials) != np.array(self.previous_materials)
        )
        reward += material_change_penalty
        
        self.previous_materials = self.fixed_materials.copy()
        
        return self.current_reflection.astype(np.float32), reward, False, False, {}

class MaterialOptimizer:
    def __init__(self, target_reflection, num_layers=8):
        self.target_reflection = target_reflection
        self.num_layers = num_layers
        self.available_materials = AVAILABLE_MATERIALS
    
    def _selection(self, population, scores):
        """选择操作"""
        selected = []
        # 使用锦标赛选择
        tournament_size = min(3, len(population))  # 确保锦标赛大小不超过种群大小
        
        for _ in range(len(population) // 2):
            # 随机选择tournament_size个个体进行比较
            tournament = random.sample(list(enumerate(scores)), tournament_size)
            winner_idx = max(tournament, key=lambda x: x[1])[0]
            selected.append(population[winner_idx])
            
            # 如果选择的个体太少，补充到原始种群大小的一半
            if len(selected) < len(population) // 2:
                remaining = len(population) // 2 - len(selected)
                # 从剩余的个体中随机选择
                remaining_indices = list(set(range(len(population))) - set([x[0] for x in tournament]))
                if remaining_indices:
                    additional = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
                    selected.extend([population[idx] for idx in additional])
        
        return selected

    def _crossover(self, population):
        """交叉操作"""
        if len(population) < 2:  # 如果种群太小，直接复制
            return population * 2
        
        new_population = []
        target_size = len(population)  # 目标种群大小
        
        while len(new_population) < target_size:
            # 如果剩余的位置只够放一个个体
            if len(new_population) == target_size - 1:
                new_population.append(random.choice(population))
                break
            
            # 从现有种群中选择两个父代
            available_parents = list(range(len(population)))
            parent1_idx = random.choice(available_parents)
            available_parents.remove(parent1_idx)
            parent2_idx = random.choice(available_parents)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # 单点交叉
            crossover_point = random.randint(1, self.num_layers - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            new_population.extend([child1, child2])
        
        return new_population[:target_size]  # 确保返回正确的种群大小

    def _mutation(self, population, materials):
        """变异操作"""
        mutation_rate = 0.1  # 变异概率
        for i in range(len(population)):
            if random.random() < mutation_rate:
                # 随机选择一个位置进行变异
                mutation_point = random.randint(0, self.num_layers - 1)
                # 随机选择一个新材料
                new_material = random.choice(materials)
                # 应用变异
                population[i][mutation_point] = new_material
        return population

    def _merge_adjacent_same_materials(self, sequence, thicknesses=None):
        """合并相邻的相同材料层"""
        if len(sequence) <= 1:
            return sequence, thicknesses
            
        merged_sequence = []
        merged_thicknesses = None
        if thicknesses is not None:
            merged_thicknesses = []
            
        i = 0
        while i < len(sequence):
            current_material = sequence[i]
            current_thickness = thicknesses[i] if thicknesses is not None else None
            
            # 查找连续的相同材料
            j = i + 1
            while j < len(sequence) and sequence[j] == current_material:
                if thicknesses is not None:
                    current_thickness += thicknesses[j]
                j += 1
                
            # 添加到合并后的列表
            merged_sequence.append(current_material)
            if merged_thicknesses is not None:
                merged_thicknesses.append(current_thickness)
                
            # 移动到下一个不同的材料
            i = j
            
        return merged_sequence, merged_thicknesses

    def optimize_materials(self, population_size=10, generations=5):
        """使用遗传算法优化材料序列"""
        materials = list(self.available_materials.keys())
        population = []
        
        # 生成初始种群
        for _ in range(population_size):
            sequence = [random.choice(materials) for _ in range(self.num_layers)]
            population.append(sequence)
            
        best_sequence = None
        best_score = float('-inf')
        
        # 遗传算法主循环
        for generation in range(generations):
            # 评估种群
            scores = []
            for sequence in population:
                score = self._evaluate_material_sequence(sequence)
                scores.append(score)
                
                # 更新最佳序列
                if score > best_score:
                    best_score = score
                    # 合并相邻相同材料后再保存
                    merged_sequence, _ = self._merge_adjacent_same_materials(sequence.copy())
                    best_sequence = merged_sequence
            
            print(f"Generation {generation + 1}, Best score: {best_score}")
            print(f"Best sequence: {best_sequence}")
            
            # 选择
            selected = self._selection(population, scores)
            # 交叉
            new_population = self._crossover(selected)
            # 变异
            new_population = self._mutation(new_population, materials)
            
            population = new_population
        
        # 最终返回前再次确保相邻相同材料已合并
        if best_sequence:
            best_sequence, _ = self._merge_adjacent_same_materials(best_sequence)
            
        return best_sequence

    def _evaluate_material_sequence(self, sequence):
        """评估材料序列的性能"""
        # 首先合并相邻相同材料
        merged_sequence, _ = self._merge_adjacent_same_materials(sequence)
        
        # 创建临时环境
        env = ThicknessOptimizationEnv(
            target_reflection=self.target_reflection,
            materials_sequence=merged_sequence
        )
        
        # 进行多次随机厚度评估
        trials = 5
        total_score = 0
        
        for _ in range(trials):
            obs, _ = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 50:
                action = env.action_space.sample()  # 随机动作
                obs, reward, done, _, _ = env.step(action)
                total_score += reward
                step_count += 1
                
        return total_score / trials

if __name__ == "__main__":
    # 检查Python环境
    import sys
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # 检查必要包的版本
    import gymnasium
    import stable_baselines3
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
        best_mse, best_reward, best_thicknesses = evaluate_model(best_model, env, num_episodes=10,render=False)
        
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
        import traceback
        traceback.print_exc() 