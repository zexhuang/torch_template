import os
import yaml

CONFIG_PATH = 'config/'

def load_config():
    with open(os.path.join(CONFIG_PATH, 'config.yaml')) as file:
        config = yaml.safe_load(file)

    return config
    
