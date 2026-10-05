# 1. Hardware Information

All experiments are conducted on a single `NVIDIA A100 Tensor Core GPU`.

# 2. Data Preparation

The code and instructions for data preprocessing are located in the `DataProcessProject` folder. Please read the `README` file in that folder and follow the steps to complete the preprocessing. Do not proceed to the next step until the data preparation is successfully verified.

# 3. Virtual Environment Setup

We provide two virtual environment configurations to accommodate different experimental requirements:

- **Time-series Foundation Models**: See the `TS-LibProject` folder. **Python 3.8 is required**.
- **Time-series Large Models**: See the `LLMProject` folder. **Python 3.10 is required**.

The specific dependency information (`requirements.txt`) can be found in each corresponding subfolder.

We recommend using `conda` for virtual environment management, as it greatly simplifies the setup process. Please refer to external documentation for detailed instructions on how to use `conda`.

# 4. Reproduction Guide

The model reproduction is divided into two main parts:

- **Part 1: Time-series Foundation Models** (see the `TS-LibProject` folder)
- **Part 2: Time-series Large Models** (see the `LLMProject` folder)

Detailed reproduction steps for each part can be found in the `README.md` file within the corresponding folder.
