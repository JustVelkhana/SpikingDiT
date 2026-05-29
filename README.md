# Trajectory-Aware Spiking DiTs Conversion via Membrane Potential Error-Feedback

This is the code implementation for the paper "Trajectory-Aware Spiking DiTs Conversion via Membrane Potential Error-Feedback" in ICML 2026.

![](main.png)

Environment Setup

The project requires the `fast-hadamard-transform` package along with common libraries such as `numpy`, `torch`, `diffusers`, `tqdm`, and `datasets`.

- **Basic Setup**: Initialize the environment using `environment.yml`.
- **Install `fast-hadamard-transform`**:
```bash
git clone [https://github.com/Dao-AILab/fast-hadamard-transform.git](https://github.com/Dao-AILab/fast-hadamard-transform.git)
# or: git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e . --no-build-isolation
```
Evaluation Environment: Create a separate environment for metric evaluation:
```
conda create -n evaluator python=3.9
conda activate evaluator
pip install tensorflow-gpu==2.10.0
pip install scipy tqdm requests
```

### 2. Running the Code
Execute the main script using the following command:
```
bash main.sh <gpu> <T> <n> <steps> <image_size>
```

Configurable parameters: T=4, n=6/8/10, steps=25/50/100, image_size=256/512
