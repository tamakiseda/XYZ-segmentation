"""
@Create time : 2023/12/29 16:30
@Functions   :
@Author      : Lixiang
"""
import os
import json


def read_config(config_path):
    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def write_config(config_path, config):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    ...
