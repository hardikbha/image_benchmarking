"""
setup.py for DFToolBench-I

DFToolBench-I: Benchmarking Tool-Augmented Agents for Image-Based Deepfake Detection
Published in IEEE Transactions on Information Forensics and Security.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

# Core runtime dependencies (minimal set needed to import the package and run tools)
INSTALL_REQUIRES = [
    # Deep learning core
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",

    # Vision & image I/O
    "Pillow>=9.0.0",
    "opencv-python>=4.7.0",
    "imageio>=2.28.0",

    # Numerical / scientific
    "numpy>=1.23.0",
    "scipy>=1.10.0",

    # OCR
    "easyocr>=1.7.0",

    # Transformers & vision models
    "transformers>=4.38.0",
    "timm>=0.9.0",
    "einops>=0.6.0",

    # Object detection
    "ultralytics>=8.0.0",

    # CLIP (used by DeepfakeDetectionTool / D3)
    "open-clip-torch>=2.20.0",

    # HTTP server (tool server)
    "flask>=3.0.0",
    "requests>=2.31.0",
    "gunicorn>=21.0.0",

    # Data handling
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "jsonschema>=4.17.0",

    # Misc utilities
    "rich>=13.0.0",
    "click>=8.1.0",
]

# Optional dependencies grouped by use-case
EXTRAS_REQUIRE = {
    # LLM evaluation — API clients for OpenAI, Anthropic, Google
    "eval": [
        "openai>=1.30.0",
        "anthropic>=0.25.0",
        "google-generativeai>=0.5.0",
        "tiktoken>=0.6.0",
        "sentence-transformers>=2.7.0",
        "scikit-learn>=1.4.0",
        # OpenCompass evaluation framework
        "opencompass>=0.2.0",
    ],

    # Local / open-weight LLM inference
    "local-llm": [
        "vllm>=0.4.0",
        "accelerate>=0.29.0",
        "bitsandbytes>=0.43.0",
    ],

    # Development & testing
    "dev": [
        "pytest>=8.0.0",
        "pytest-cov>=5.0.0",
        "black>=24.0.0",
        "isort>=5.13.0",
        "flake8>=7.0.0",
        "mypy>=1.9.0",
        "pre-commit>=3.7.0",
    ],

    # Documentation
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=2.0.0",
        "myst-parser>=3.0.0",
    ],
}

# "all" convenience target installs eval + dev
EXTRAS_REQUIRE["all"] = (
    EXTRAS_REQUIRE["eval"]
    + EXTRAS_REQUIRE["local-llm"]
    + EXTRAS_REQUIRE["dev"]
)

setup(
    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #
    name="dftoolbench",
    version="1.0.0",
    description=(
        "DFToolBench-I: Benchmarking Tool-Augmented Agents for "
        "Image-Based Deepfake Detection"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",

    # ------------------------------------------------------------------ #
    # Authors & project URLs                                               #
    # ------------------------------------------------------------------ #
    author=(
        "Hardik Sharma, Prateek Shaily, Jayant Kumar, "
        "Praful Hambarde, Amit Shukla, Sachin Chaudhary"
    ),
    author_email="hardiknegiaditya@gmail.com",
    url="https://github.com/<your-org>/DFToolBench-I",
    project_urls={
        "Paper": "https://ieeexplore.ieee.org/",
        "Bug Tracker": "https://github.com/<your-org>/DFToolBench-I/issues",
        "Documentation": "https://github.com/<your-org>/DFToolBench-I#readme",
    },

    # ------------------------------------------------------------------ #
    # Licence & classifiers                                                #
    # ------------------------------------------------------------------ #
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Security",
    ],

    # ------------------------------------------------------------------ #
    # Package discovery                                                    #
    # ------------------------------------------------------------------ #
    packages=find_packages(exclude=["tests*", "scripts*", "examples*"]),
    python_requires=">=3.9",

    # ------------------------------------------------------------------ #
    # Dependencies                                                         #
    # ------------------------------------------------------------------ #
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # ------------------------------------------------------------------ #
    # CLI entry points                                                     #
    # ------------------------------------------------------------------ #
    entry_points={
        "console_scripts": [
            "dftoolbench-server=dftoolbench.utils.tool_server:main",
            "dftoolbench-eval=dftoolbench.evaluation.run:main",
            "dftoolbench-smoke=scripts.smoke_test:main",
        ],
    },

    # ------------------------------------------------------------------ #
    # Package data                                                         #
    # ------------------------------------------------------------------ #
    include_package_data=True,
    package_data={
        "dftoolbench": [
            "data/schemas/*.json",
            "evaluation/rubrics/*.yaml",
        ],
    },
)
