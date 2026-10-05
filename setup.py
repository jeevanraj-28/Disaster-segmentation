from setuptools import find_packages, setup

setup(
    name="disaster-segmentation",
    version="1.0.0",
    description="U-Net (ResNet34) semantic segmentation of FloodNet drone imagery",
    author="Jeevan Raj M",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10,<3.12",
)
