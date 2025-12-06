"""Setup script for tourism prediction project"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tourism-prediction",
    version="1.0.0",
    author="Tourism Prediction Team",
    description="ML pipeline for predicting tourism demand levels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rahul-ig/tourism-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.7.0",
        "huggingface-hub>=0.17.0",
        "datasets>=2.14.0",
    ],
)
