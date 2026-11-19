from setuptools import find_packages, setup

setup(
    name="tweet_topic",
    version="1.0.0",
    description="Modular and Extensible Multi-Label Topic Classification Benchmark",
    author="CardiffNLP Tweet Topic Research",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "datasets",
        "transformers",
        "torch",
        "pyyaml",
    ],
)
