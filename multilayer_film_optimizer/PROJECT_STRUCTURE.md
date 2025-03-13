# 项目结构

```
multilayer_film_optimizer/
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── main.py                   # 主程序入口
├── PROJECT_STRUCTURE.md      # 项目结构说明（本文件）
├── core/                     # 核心功能模块
│   ├── __init__.py
│   ├── environment.py        # 优化环境
│   ├── materials.py          # 材料库管理
│   └── optimizer.py          # 优化器
├── config/                   # 配置文件
│   ├── __init__.py
│   └── default_config.py     # 默认配置
├── data/                     # 数据文件
│   ├── __init__.py
│   └── materials/            # 材料数据
│       └── materials_config.json  # 材料配置
├── models/                   # 模型存储
│   └── __init__.py
├── utils/                    # 工具函数
│   ├── __init__.py
│   └── visualization.py      # 可视化工具
├── ui/                       # 用户界面
│   └── __init__.py
└── examples/                 # 示例脚本
    ├── optimize_ir_filter.py # 优化红外滤波器示例
    └── custom_target.py      # 自定义目标反射率示例
```

## 模块说明

### core/ - 核心功能模块

- `environment.py`: 定义了优化环境，包括薄膜层的设置、反射率计算等
- `materials.py`: 材料库管理，包括材料加载、材料属性获取等
- `optimizer.py`: 优化器，使用强化学习算法优化薄膜厚度

### config/ - 配置文件

- `default_config.py`: 默认配置，包括默认的目标反射率、波长范围等

### data/ - 数据文件

- `materials/`: 材料数据目录，存放材料的折射率数据文件

### utils/ - 工具函数

- `visualization.py`: 可视化工具，用于绘制反射率曲线和优化结果

### examples/ - 示例脚本

- `optimize_ir_filter.py`: 优化红外滤波器示例
- `custom_target.py`: 自定义目标反射率示例

## 主要类和函数

### ThicknessOptimizationEnv

薄膜厚度优化环境，用于优化多层薄膜的厚度，以达到特定的光学反射率目标。

### MaterialLibrary

材料库类，用于管理和加载材料。

### FilmOptimizer

薄膜优化器类，用于训练和评估模型。

### Visualizer

可视化工具类，用于绘制反射率曲线和优化结果。 