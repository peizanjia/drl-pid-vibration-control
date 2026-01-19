import os
import yaml
import numpy as np
from scipy import io


def load_config(file_name='dynamics_model_config.yaml'):
    """
    加载位于本文件（config/config.py）同目录下的 YAML 配置文件。
    该函数假设它位于 config/ 目录下。
    """

    # 1. 确定本文件（config.py）的绝对路径
    # current_dir 此时指向 DRL_PID_PROJECT/config/
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 拼接配置文件的完整路径
    config_path = os.path.join(current_dir, file_name)

    print(f"尝试从路径加载配置: {config_path}")  # 增加调试信息

    try:
        # 3. 加载并返回
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # 4. 提供清晰的错误提示
        raise FileNotFoundError(
            f"配置文件未找到！请检查文件是否存在于 'config/' 目录下，"
            f"期望路径为: {config_path}"
        )


def load_mat(file_name='BendingMomentResult.mat', variable_name='Mt'):
    """
    读取 MATLAB .mat 文件并转换为 ndarray。

    Args:
        file_name (str): 文件名（若在 config 目录下）或绝对路径。
        variable_name (str, optional): 指定读取的变量名。若为 None，则返回整个字典。

    Returns:
        np.ndarray or dict: 返回提取的矩阵数据。
    """
    # 自动兼容：如果是相对路径，则默认去 config 目录下找
    if not os.path.isabs(file_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(base_path, file_name)

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Matlab 数据文件未找到: {file_name}")

    try:
        # mat_data 是一个字典
        mat_data = io.loadmat(file_name)

        # 排除 matlab 自动生成的元数据标记
        clean_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

        if variable_name:
            if variable_name in clean_data:
                return np.array(clean_data[variable_name])
            else:
                available_keys = list(clean_data.keys())
                raise KeyError(f"变量 '{variable_name}' 不存在。可用变量: {available_keys}")

        return clean_data

    except Exception as e:
        raise IOError(f"处理 .mat 文件时发生异常: {str(e)}")