import numpy as np
import gymnasium
import tmm_fast.gym_multilayerthinfilm as mltf
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from .materials import MaterialLibrary

class ThicknessOptimizationEnv(gymnasium.Env):
    """
    薄膜厚度优化环境
    
    该环境用于优化多层薄膜的厚度，以达到特定的光学反射率目标
    """
    
    def __init__(self, target_reflection, materials_sequence=None, num_layers=8, 
                 min_thickness=5e-9, max_thickness=500e-9, wavelength_range=(380, 14000, 1363)):
        """
        初始化环境
        
        参数:
            target_reflection: 目标反射率曲线
            materials_sequence: 材料序列，如果为None则自动选择
            num_layers: 薄膜层数
            min_thickness: 最小厚度 (m)
            max_thickness: 最大厚度 (m)
            wavelength_range: 波长范围 (nm_min, nm_max, points)
        """
        self.num_layers = num_layers
        self.target_reflection = target_reflection
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        
        # 设置波长范围
        wl_min, wl_max, wl_points = wavelength_range
        self.wl = np.linspace(wl_min, wl_max, wl_points) * (10**(-9))
        
        # 加载材料库
        self.material_library = MaterialLibrary()
        self.available_materials = self.material_library.get_available_materials()
        self.material_paths = self.material_library.get_material_paths()
        
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
            # 使用指定的材料序列
            self.materials_sequence = materials_sequence
            
        # 更新fixed_materials为材料索引
        self.fixed_materials = [list(self.available_materials.keys()).index(mat) for mat in self.materials_sequence]
        
        # 创建目标反射率数组
        target_array = np.zeros((1, len(self.wl)))
        
        # 使用传入的target_reflection设置目标反射率
        for wavelength_range in self.target_reflection["wavelength_ranges"]:
            range_start, range_end = wavelength_range["range"]
            target_value = wavelength_range["target"]
            target_array[0, (self.wl >= range_start) & (self.wl <= range_end)] = target_value
        
        self.target_array = target_array
        
        # 定义关注的波长范围
        self.high_reflection_ranges = []
        for wavelength_range in self.target_reflection["wavelength_ranges"]:
            if wavelength_range["target"] > 0.5:  # 高反射区域
                self.high_reflection_ranges.append(wavelength_range["range"])
        
        # 设置权重
        weights = np.ones_like(target_array)
        # 为每个高反射区域设置权重
        for range_start, range_end in self.high_reflection_ranges:
            weights[0, (self.wl >= range_start) & (self.wl <= range_end)] = 5.0
        
        # 创建target字典
        self.target = {
            'direction': np.array([0]),  # 固定入射角为0度
            'spectrum': self.wl,
            'target': target_array,
            'mode': 'reflectivity'
        }
        
        # 创建基础环境
        self.env = mltf.MultiLayerThinFilm(
            self.N, 
            self.num_layers,
            self.target,
            weights=weights,
            max_thickness=self.max_thickness,
            min_thickness=self.min_thickness,
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
        
        # 初始化当前厚度
        self.current_thicknesses = np.zeros(self.num_layers)
        self.current_reflection = np.zeros(len(self.wl))
        self.best_thicknesses = None
        self.best_reflection = None
        self.best_reward = float('-inf')

    def _optimize_material_sequence(self):
        """
        使用遗传算法优化材料序列
        
        返回:
            最优材料序列
        """
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
                for wavelength_range in self.target_reflection["wavelength_ranges"]:
                    range_start, range_end = wavelength_range["range"]
                    target_value = wavelength_range["target"]
                    mask = (self.wl >= range_start) & (self.wl <= range_end)
                    achieved = reflection[mask]
                    target = np.ones_like(achieved) * target_value
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
                best_sequence = population[max_score_idx]
            
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
        """
        重置环境
        
        参数:
            seed: 随机种子
            
        返回:
            初始观察和信息
        """
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
        """
        执行一步优化
        
        参数:
            action: 动作，表示每层的厚度
            
        返回:
            观察、奖励、是否结束、是否截断、信息
        """
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
        
        # 更新最佳结果
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_thicknesses = self.current_thicknesses.copy()
            self.best_reflection = self.current_reflection.copy()
        
        # 始终返回False，因为我们希望持续优化
        done = False
        
        # 确保返回的observation是float32类型
        return self.current_reflection.astype(np.float32), reward, done, False, {}

    def _get_reflection(self):
        """
        获取当前配置的反射率
        
        返回:
            当前反射率
        """
        # 重置环境
        self.env.reset()
        
        # 按固定顺序添加层
        for material_idx, thickness in zip(self.fixed_materials, self.current_thicknesses):
            action = self.env.create_action(material_idx + 1, thickness, is_normalized=False)
            self.env.step(action)
        
        # 确保返回float32类型
        return self.env.simulation[0].astype(np.float32)

    def _calculate_reward(self):
        """
        计算奖励，使用软性惩罚
        
        返回:
            奖励值
        """
        base_reward, _ = self.env.reward_func(
            self.current_reflection.reshape(1, -1),
            self.target['target'],
            self.env.weights,
            self.env.baseline_mse,
            self.env.normalization
        )
        
        # 计算关键区域的匹配度
        critical_penalty = 0
        
        # 对每个波长范围计算惩罚
        for wavelength_range in self.target_reflection["wavelength_ranges"]:
            range_start, range_end = wavelength_range["range"]
            target_value = wavelength_range["target"]
            mask = (self.wl >= range_start) & (self.wl <= range_end)
            
            # 计算该区域的误差
            error = np.mean(np.abs(
                self.current_reflection[mask] - target_value
            ))
            
            # 只有当误差大于阈值时才添加惩罚
            threshold = 0.05
            if error > threshold:
                # 高反射区域的惩罚更大
                if target_value > 0.5:
                    critical_penalty -= 0.2 * error
                else:
                    critical_penalty -= 0.1 * error
        
        return base_reward + critical_penalty

    def render(self):
        """
        渲染当前结果
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.wl * 1e6, self.target_array[0], 'b-', label='Target')
        plt.plot(self.wl * 1e6, self.current_reflection, 'r--', label='Current')
        
        if self.best_reflection is not None:
            plt.plot(self.wl * 1e6, self.best_reflection, 'g-.', label='Best')
            
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectivity')
        plt.title('Reflectivity vs Wavelength')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def save_results(self, output_dir="results"):
        """
        保存优化结果
        
        参数:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存最佳厚度
        if self.best_thicknesses is not None:
            result = {
                "materials": self.materials_sequence,
                "thicknesses": [float(t) for t in self.best_thicknesses],
                "thicknesses_nm": [float(t * 1e9) for t in self.best_thicknesses],
                "reward": float(self.best_reward),
                "wavelength": {
                    "min": float(self.wl.min()),
                    "max": float(self.wl.max()),
                    "points": len(self.wl)
                },
                "target_reflection": self.target_reflection
            }
            
            with open(os.path.join(output_dir, "optimization_result.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            # 保存反射率图
            plt.figure(figsize=(10, 6))
            plt.plot(self.wl * 1e6, self.target_array[0], 'b-', label='Target')
            plt.plot(self.wl * 1e6, self.best_reflection, 'r--', label='Optimized')
            plt.xlabel('Wavelength (μm)')
            plt.ylabel('Reflectivity')
            plt.title('Reflectivity Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "reflection_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            return result
        else:
            print("没有可保存的最佳结果")
            return None
