import random
import numpy as np
from materials_config import AVAILABLE_MATERIALS
from modules.env import ThicknessOptimizationEnv

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