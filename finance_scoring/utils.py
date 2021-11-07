import yaml
from easydict import EasyDict as edict

def load_config(config_yml="config.yml"):

    with open(config_yml, "r") as f:
        return edict(yaml.load(f, Loader=yaml.FullLoader))