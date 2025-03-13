"""
默认配置文件
包含默认的目标反射率、波长范围等配置
"""

# 默认波长范围 (nm)
DEFAULT_WAVELENGTH_RANGE = (380, 14000, 1363)

# 默认厚度范围 (m)
DEFAULT_THICKNESS_RANGE = {
    "min": 5e-9,   # 5 nm
    "max": 500e-9  # 500 nm
}

# 默认目标反射率配置
DEFAULT_TARGET_REFLECTION = {
    "wavelength_ranges": [
        {"range": [0.38e-6, 0.8e-6], "target": 0.0},  # 可见光区域低反射
        {"range": [3e-6, 5e-6], "target": 1.0},       # 中红外高反射
        {"range": [5e-6, 8e-6], "target": 0.0},       # 中红外低反射
        {"range": [8e-6, 14e-6], "target": 1.0}       # 远红外高反射
    ]
}

# 默认训练配置
DEFAULT_TRAINING_CONFIG = {
    "total_timesteps": 10000,
    "learning_rate": 1e-3,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10
}

# 默认评估配置
DEFAULT_EVALUATION_CONFIG = {
    "num_episodes": 10,
    "render": True
}

# 默认材料序列
DEFAULT_MATERIALS_SEQUENCE = ["W", "Ge", "SiO2", "W", "Ge"]

# 预设目标反射率配置
PRESET_TARGETS = {
    "visible_ir_filter": {
        "name": "可见光-红外滤波器",
        "description": "可见光透过，红外反射的滤波器",
        "target_reflection": {
            "wavelength_ranges": [
                {"range": [0.38e-6, 0.8e-6], "target": 0.0},  # 可见光区域低反射
                {"range": [0.8e-6, 14e-6], "target": 1.0}     # 红外区域高反射
            ]
        }
    },
    "ir_bandpass": {
        "name": "红外带通滤波器",
        "description": "特定红外波段透过，其他波段反射的滤波器",
        "target_reflection": {
            "wavelength_ranges": [
                {"range": [0.38e-6, 3e-6], "target": 1.0},    # 短波区域高反射
                {"range": [3e-6, 5e-6], "target": 0.0},       # 中红外透过
                {"range": [5e-6, 14e-6], "target": 1.0}       # 长波区域高反射
            ]
        }
    },
    "heat_mirror": {
        "name": "热反射镜",
        "description": "可见光透过，热辐射反射的镜面",
        "target_reflection": {
            "wavelength_ranges": [
                {"range": [0.38e-6, 0.8e-6], "target": 0.0},  # 可见光区域低反射
                {"range": [0.8e-6, 3e-6], "target": 0.5},     # 近红外部分反射
                {"range": [3e-6, 14e-6], "target": 0.9}       # 中远红外高反射
            ]
        }
    },
    "dual_band": {
        "name": "双波段滤波器",
        "description": "两个特定波段透过，其他波段反射的滤波器",
        "target_reflection": {
            "wavelength_ranges": [
                {"range": [0.38e-6, 1e-6], "target": 0.0},    # 可见光区域低反射
                {"range": [1e-6, 3e-6], "target": 1.0},       # 近红外高反射
                {"range": [3e-6, 5e-6], "target": 0.0},       # 中红外低反射
                {"range": [5e-6, 14e-6], "target": 1.0}       # 远红外高反射
            ]
        }
    },
    "custom": {
        "name": "自定义",
        "description": "用户自定义的目标反射率",
        "target_reflection": DEFAULT_TARGET_REFLECTION
    }
}
