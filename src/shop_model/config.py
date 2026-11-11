import os
import yaml

def load_config(path=None):
    path = path or os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(path) as f:
        config = yaml.safe_load(f)
    
    return config