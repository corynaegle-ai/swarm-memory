from setuptools import setup, find_packages

setup(
    name="swarm-memory",
    version="1.0.0",
    description="Memory extraction pipeline with Jetson offload for cost reduction",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "aiohttp>=3.8.0",
        "tenacity>=8.0.0",
        "anthropic>=0.20.0",
        "requests>=2.28.0",
    ],
    python_requires=">=3.9",
)
