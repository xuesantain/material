import matplotlib.pyplot as plt
import numpy as np
import os
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Visualizer:
    """可视化工具类，用于绘制反射率曲线和优化结果"""
    
    @staticmethod
    def plot_reflection(wavelength, reflection, target=None, title="Reflectivity vs Wavelength", 
                        save_path=None, show=True):
        """
        绘制反射率曲线
        
        参数:
            wavelength: 波长数组 (m)
            reflection: 反射率数组
            target: 目标反射率数组 (可选)
            title: 图表标题
            save_path: 保存路径 (可选)
            show: 是否显示图表
            
        返回:
            matplotlib图表对象
        """
        fig = plt.figure(figsize=(10, 6))
        
        # 将波长转换为微米
        wavelength_um = wavelength * 1e6
        
        # 绘制反射率曲线
        plt.plot(wavelength_um, reflection, 'r-', label='Reflection')
        
        # 如果有目标反射率，也绘制出来
        if target is not None:
            plt.plot(wavelength_um, target, 'b--', label='Target')
            
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectivity')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    @staticmethod
    def plot_optimization_progress(log_file, metrics=None, save_path=None, show=True):
        """
        绘制优化进度
        
        参数:
            log_file: 日志文件路径
            metrics: 要绘制的指标列表 (可选)
            save_path: 保存路径 (可选)
            show: 是否显示图表
            
        返回:
            matplotlib图表对象
        """
        # 默认指标
        if metrics is None:
            metrics = ['reward', 'mse', 'similarity']
            
        # 加载日志数据
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # 解析日志数据
        steps = []
        data = {metric: [] for metric in metrics}
        
        for line in lines:
            if line.strip():
                try:
                    parts = line.strip().split(',')
                    step = int(parts[0])
                    steps.append(step)
                    
                    for part in parts[1:]:
                        name, value = part.split(':')
                        name = name.strip()
                        if name in metrics:
                            data[name].append(float(value))
                except:
                    continue
        
        # 创建图表
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
        
        # 如果只有一个指标，将axes转换为列表
        if len(metrics) == 1:
            axes = [axes]
            
        # 绘制每个指标
        for i, metric in enumerate(metrics):
            if metric in data and len(data[metric]) > 0:
                axes[i].plot(steps[:len(data[metric])], data[metric])
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
                
        # 设置x轴标签
        axes[-1].set_xlabel('Steps')
        
        # 设置标题
        plt.suptitle('Optimization Progress')
        plt.tight_layout()
        
        # 保存图表
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    @staticmethod
    def plot_layer_structure(materials, thicknesses, save_path=None, show=True):
        """
        绘制薄膜层结构
        
        参数:
            materials: 材料列表
            thicknesses: 厚度列表 (nm)
            save_path: 保存路径 (可选)
            show: 是否显示图表
            
        返回:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 颜色映射
        colors = {
            'W': '#808080',      # 灰色
            'Ge': '#A52A2A',     # 棕色
            'SiO2': '#ADD8E6',   # 浅蓝色
            'Si': '#0000FF',     # 蓝色
            'Al': '#C0C0C0',     # 银色
            'Au': '#FFD700',     # 金色
            'Ag': '#C0C0C0',     # 银色
            'Cu': '#B87333',     # 铜色
            'Ti': '#808080',     # 灰色
            'TiO2': '#FFFFFF',   # 白色
        }
        
        # 默认颜色
        default_color = '#CCCCCC'
        
        # 计算总厚度
        total_thickness = sum(thicknesses)
        
        # 绘制每一层
        y_pos = 0
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            color = colors.get(material, default_color)
            ax.barh(0, thickness, left=y_pos, height=1, color=color, edgecolor='black')
            
            # 添加材料标签
            if thickness > total_thickness * 0.05:  # 只在较厚的层上添加标签
                ax.text(y_pos + thickness/2, 0, f"{material}\n{thickness:.1f} nm", 
                        ha='center', va='center', fontsize=10)
                
            y_pos += thickness
            
        # 设置坐标轴
        ax.set_xlim(0, total_thickness)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Thickness (nm)')
        ax.set_yticks([])
        
        # 设置标题
        plt.title('Multilayer Film Structure')
        
        # 保存图表
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    @staticmethod
    def create_report(result_file, output_dir="reports"):
        """
        创建优化结果报告
        
        参数:
            result_file: 结果文件路径
            output_dir: 输出目录
            
        返回:
            报告文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载结果数据
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            
        # 提取数据
        materials = result["materials"]
        thicknesses_nm = result["thicknesses_nm"]
        wavelength_range = result["wavelength"]
        target_reflection = result["target_reflection"]
        
        # 创建报告文件名
        report_name = os.path.splitext(os.path.basename(result_file))[0] + "_report.html"
        report_path = os.path.join(output_dir, report_name)
        
        # 创建层结构图
        structure_fig = Visualizer.plot_layer_structure(materials, thicknesses_nm, show=False)
        structure_img_path = os.path.join(output_dir, "structure.png")
        structure_fig.savefig(structure_img_path, dpi=300, bbox_inches='tight')
        
        # 创建HTML报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>多层薄膜优化结果报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .container {{ display: flex; justify-content: center; margin: 20px 0; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>多层薄膜优化结果报告</h1>
                
                <h2>材料和厚度配置</h2>
                <table>
                    <tr>
                        <th>层号</th>
                        <th>材料</th>
                        <th>厚度 (nm)</th>
                    </tr>
            """)
            
            # 添加材料和厚度表格
            for i, (material, thickness) in enumerate(zip(materials, thicknesses_nm)):
                f.write(f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{material}</td>
                        <td>{thickness:.2f}</td>
                    </tr>
                """)
                
            f.write("""
                </table>
                
                <h2>层结构图</h2>
                <div class="container">
                    <img src="structure.png" alt="Layer Structure">
                </div>
                
                <h2>目标反射率</h2>
                <table>
                    <tr>
                        <th>波长范围 (μm)</th>
                        <th>目标反射率</th>
                    </tr>
            """)
            
            # 添加目标反射率表格
            for wavelength_range in target_reflection["wavelength_ranges"]:
                start, end = wavelength_range["range"]
                target = wavelength_range["target"]
                f.write(f"""
                    <tr>
                        <td>{start*1e6:.2f} - {end*1e6:.2f}</td>
                        <td>{target:.2f}</td>
                    </tr>
                """)
                
            f.write("""
                </table>
                
                <h2>优化结果</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                    </tr>
                    <tr>
                        <td>奖励值</td>
                        <td>{:.4f}</td>
                    </tr>
                </table>
                
                <div class="container">
                    <img src="../reflection_comparison.png" alt="Reflection Comparison">
                </div>
                
                <p>报告生成时间: {}</p>
            </body>
            </html>
            """.format(result["reward"], import_time().strftime("%Y-%m-%d %H:%M:%S")))
            
        return report_path

def import_time():
    """导入时间模块并返回当前时间"""
    import time
    from datetime import datetime
    return datetime.now()
