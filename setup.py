from pathlib import Path
from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="chara-survival",
    version="0.1.9",
    description="Thermodynamic Graph Laplacian survival inference for out-of-distribution transcriptomic oncology",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Sharon Melhi",
    author_email="sharonmelhi365@gmail.com",
    license="MIT",
    url="https://github.com/Sharon-codes/Chara",
    project_urls={
        "Hugging Face Hub": "https://huggingface.co/SharonMelhi/chara-survival",
        "Web App": "https://chara-frontend.vercel.app",
        "PyPI": "https://pypi.org/project/chara-survival/",
        "Bug Tracker": "https://github.com/Sharon-codes/Chara/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.1",
        "scikit-learn==1.5.2",
        "scikit-survival==0.23.1",
        "lifelines==0.28.0",
        "joblib==1.3.2",
        "huggingface_hub>=0.20.0",
    ],
    keywords=[
        "survival analysis",
        "cancer genomics",
        "graph laplacian",
        "molecular dynamics",
        "out-of-distribution generalization",
        "transcriptomics",
        "lung adenocarcinoma",
        "CoxNet",
        "huggingface",
    ],
)
