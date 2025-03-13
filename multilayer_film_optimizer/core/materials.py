import os
import json
import numpy as np
from pathlib import Path

class MaterialLibrary:
    """材料库类，用于管理和加载材料"""
    
    def __init__(self, materials_dir=None):
        """
        初始化材料库
        
        参数:
            materials_dir: 材料数据目录，如果为None则使用默认目录
        """
        if materials_dir is None:
            # 使用默认材料目录
            self.materials_dir = Path(__file__).parent.parent / "data" / "materials"
        else:
            self.materials_dir = Path(materials_dir)
            
        # 确保材料目录存在
        os.makedirs(self.materials_dir, exist_ok=True)
        
        # 加载材料配置
        self.materials_config = {}
        self._load_materials_config()
        
    def _load_materials_config(self):
        """加载材料配置"""
        config_path = self.materials_dir / "materials_config.json"
        
        # 如果配置文件不存在，创建默认配置
        if not config_path.exists():
            default_config = {
                "W": {
                    "name": "Tungsten",
                    "path": str(self.materials_dir / "W.nk"),
                    "description": "钨材料"
                },
                "Ge": {
                    "name": "Germanium",
                    "path": str(self.materials_dir / "Ge.nk"),
                    "description": "锗材料"
                },
                "SiO2": {
                    "name": "Silicon Dioxide",
                    "path": str(self.materials_dir / "SiO2.nk"),
                    "description": "二氧化硅材料"
                }
            }
            
            # 保存默认配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
                
            self.materials_config = default_config
        else:
            # 加载现有配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self.materials_config = json.load(f)
    
    def get_available_materials(self):
        """获取所有可用材料"""
        return self.materials_config
    
    def get_material_paths(self):
        """获取所有材料的路径"""
        return [mat_info["path"] for mat_info in self.materials_config.values()]
    
    def add_material(self, material_id, name, path, description=""):
        """
        添加新材料到库中
        
        参数:
            material_id: 材料ID
            name: 材料名称
            path: 材料数据文件路径
            description: 材料描述
        """
        if material_id in self.materials_config:
            raise ValueError(f"材料ID '{material_id}' 已存在")
            
        # 添加新材料
        self.materials_config[material_id] = {
            "name": name,
            "path": str(path),
            "description": description
        }
        
        # 保存更新后的配置
        config_path = self.materials_dir / "materials_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.materials_config, f, indent=4, ensure_ascii=False)
    
    def remove_material(self, material_id):
        """
        从库中移除材料
        
        参数:
            material_id: 要移除的材料ID
        """
        if material_id not in self.materials_config:
            raise ValueError(f"材料ID '{material_id}' 不存在")
            
        # 移除材料
        del self.materials_config[material_id]
        
        # 保存更新后的配置
        config_path = self.materials_dir / "materials_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.materials_config, f, indent=4, ensure_ascii=False)
    
    def get_material_info(self, material_id):
        """
        获取材料信息
        
        参数:
            material_id: 材料ID
        
        返回:
            材料信息字典
        """
        if material_id not in self.materials_config:
            raise ValueError(f"材料ID '{material_id}' 不存在")
            
        return self.materials_config[material_id]
