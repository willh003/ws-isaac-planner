from setuptools import setup, find_packages

setup(
    name='ws-isaac-planner',
    version='0.1.0',
    author='Will Huey',
    description='Handles planning backend for mobile robot simulations',
    packages=find_packages(include=['ws_isaac_planner']),
    install_requires=[],
)