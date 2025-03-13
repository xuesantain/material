import numpy as np
import gymnasium
import tmm_fast.gym_multilayerthinfilm as mltf
import matplotlib.pyplot as plt
from copy import deepcopy
from materials_config import AVAILABLE_MATERIALS
import random
import os
import time

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