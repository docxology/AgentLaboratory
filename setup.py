from setuptools import setup, find_packages

setup(
    name="agent_lab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "networkx>=2.8.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "tiktoken>=0.5.0",
        "python-dotenv>=0.20.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "jupyter>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "agent-lab=agent_lab.main:main",
        ],
    },
    author="Agent Laboratory Contributors",
    author_email="example@example.com",
    description="LLM-powered research assistant workflow",
    keywords="LLM, research, AI, automation",
    url="https://github.com/yourusername/agentlaboratory",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 