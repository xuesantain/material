# 导入所有子模块
from modules.env import ThicknessOptimizationEnv, DynamicMaterialEnv, plot_reflection_comparison
from modules.material_optimizer import MaterialOptimizer
from modules.training import (
    train_model, 
    evaluate_model, 
    load_model_and_config, 
    is_similar_target,
    TensorboardCallback
)

# 版本信息
__version__ = "1.0.0" 