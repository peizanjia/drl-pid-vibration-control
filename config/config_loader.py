import os
import yaml
import numpy as np
from scipy import io


def load_config(file_name='dynamics_model_config.yaml'):
    """
    Load a YAML config file located next to this module (config/).
    """
    # 1) Resolve absolute path of this module
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2) Build full config path
    config_path = os.path.join(current_dir, file_name)

    print(f"Loading config from: {config_path}")

    try:
        # 3) Load and return
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # 4) Provide a clear error message
        raise FileNotFoundError(
            "Config file not found. Please check it exists under 'config/'. "
            f"Expected path: {config_path}"
        )


def load_mat(file_name='BendingMomentResult.mat', variable_name='Mt'):
    """
    Read a MATLAB .mat file and return an ndarray or dict.

    Args:
        file_name (str): File name in config/ or an absolute path.
        variable_name (str, optional): If provided, return only this variable.
            If None, return the full dict.

    Returns:
        np.ndarray or dict: Extracted data.
    """
    # If a relative path is given, default to the config/ directory
    if not os.path.isabs(file_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(base_path, file_name)

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"MATLAB data file not found: {file_name}")

    try:
        # mat_data is a dict
        mat_data = io.loadmat(file_name)

        # Remove MATLAB metadata keys
        clean_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

        if variable_name:
            if variable_name in clean_data:
                return np.array(clean_data[variable_name])
            available_keys = list(clean_data.keys())
            raise KeyError(f"Variable '{variable_name}' not found. Available: {available_keys}")

        return clean_data

    except Exception as e:
        raise IOError(f"Failed to process .mat file: {str(e)}")
