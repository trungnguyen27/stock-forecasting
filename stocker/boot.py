from global_configs import configs
import sys, os

class boot():
    def __init__(self):
        for key, path in configs.items():
            sys.path.append(path)