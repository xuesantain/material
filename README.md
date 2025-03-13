# 多层薄膜厚度优化工具

这是一个使用强化学习和遗传算法来优化多层薄膜结构的工具。它可以自动选择最佳的材料组合和厚度配置，以达到特定的光学性能要求。

## 功能特点

- **材料序列优化**：使用遗传算法自动选择最佳的材料组合序列
- **薄膜厚度优化**：使用强化学习（PPO算法）优化每一层薄膜的厚度
- **相邻相同材料合并**：自动检测并合并相邻的相同材料层
- **目标反射率定制**：可以自定义不同波长范围的目标反射率
- **可视化结果**：生成反射率对比图和详细的厚度配置报告

## 项目结构

```
.
├── main.py                    # 主程序入口
├── use_model.py               # 使用已训练模型的脚本
├── materials_config.py        # 材料配置文件
├── modules/                   # 模块目录
│   ├── __init__.py            # 模块初始化文件
│   ├── env.py                 # 环境模块（包含ThicknessOptimizationEnv类）
│   ├── material_optimizer.py  # 材料优化器模块
│   └── training.py            # 训练和评估模块
├── runingresult/              # 运行结果保存目录
└── thickness_optimization_tensorboard/ # Tensorboard日志目录
```

## 安装依赖

```bash
pip install numpy gymnasium stable-baselines3 matplotlib torch
```

## 使用方法

### 训练新模型

运行主程序来训练新模型：

```bash
python main.py
```

这将执行以下步骤：
1. 使用遗传算法优化材料序列
2. 使用PPO算法优化薄膜厚度
3. 保存最佳模型和配置
4. 生成反射率对比图和厚度配置报告

### 使用已训练模型

如果已经有训练好的模型，可以使用以下命令：

```bash
python use_model.py
```

这将加载已训练的模型，并使用它来预测最佳的薄膜厚度配置。

## 自定义目标反射率

在`main.py`或`use_model.py`中，可以修改`target_reflection`字典来自定义目标反射率：

```python
target_reflection = {
    "wavelength_ranges": [
        {"range": [0.38e-6, 0.8e-6], "target": 0.0},  # 可见光范围
        {"range": [3e-6, 5e-6], "target": 1.0},       # 中红外范围1
        {"range": [5e-6, 8e-6], "target": 0.0},       # 中红外范围2
        {"range": [8e-6, 14e-6], "target": 1.0}       # 远红外范围
    ]
}
```

## 结果示例

运行程序后，会在`runingresult`目录下生成以下文件：
- 反射率对比图（PNG格式）
- 厚度配置报告（TXT格式）

## 注意事项

- 训练过程可能需要较长时间，取决于计算机性能
- 如果有GPU可用，程序会自动使用GPU加速训练
- 可以通过修改`train_model`函数中的参数来调整训练过程 