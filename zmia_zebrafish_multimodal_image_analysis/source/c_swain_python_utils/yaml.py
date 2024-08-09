#!python3
# yaml.py

# Corban Swain, 2021

__all__ = ['read_yaml']


def read_yaml(filepath):
    import yaml

    with open(filepath, 'r') as f:
        y = yaml.safe_load(f)

    return y
