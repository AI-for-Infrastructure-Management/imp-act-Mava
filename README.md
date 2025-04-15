## Installation 

### 1. Clone repositories

To clone this repository, the JaxMARL repository and the imp-act repository, run:
```bash
git clone https://github.com/AI-for-Infrastructure-Management/imp-act-Mava.git
cd imp-act-Mava && git checkout imp_act_adaption_2
git clone https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git
cd imp-act-JaxMARL && git checkout imp_act_adaption && cd ..
git clone https://github.com/AI-for-Infrastructure-Management/imp-act.git
```

### 2. Create a virtual environment

Note that unlike imp-act-JaxMARL, we need `python=3.10` here.

Option A: Create a conda environment using the environment YAML file,
```bash
conda env create -f conda_environment.yaml
conda activate impact-mava-env
```

Option B: Create a virtual environment `impact-mava-env` using venv/poetry etc.
```bash
# create `impact-mava-env` with python=3.10
pip install poetry==1.7.1 lockfile==0.12.2
```

### 3.Install dependencies

Dependencies for all repositories have been added to the `requirements.txt` file.
You can install the dependencies and Mava using the following command:
```bash
pip install -e .
```

<details>
<summary>GPU installation</summary>
For the GPU version of jax, we need to install `jax[cuda12]`. It is easiest
to overwrite the jax installation from the above packages by running
</details>

<details>
<summary>Optional: Troubleshooting and Additional Notes</summary>

- If `cmake` is not installed, you can install it using:
    ```bash
    sudo apt-get install cmake  # For Ubuntu/Debian
    brew install cmake          # For macOS
    ```
</details>

Checkout to the `imp_act_adaption` branch since it contains the 
`roadenv_wrapper` in the JaxMARL format which we will use to interface 
imp-act with Mava. We only install JaxMARL and no dependencies.
```bash
cd imp-act-JaxMARL && git checkout imp_act_adaption
pip install --no-deps -e ".[algs]"
```

Checkout to the `99-updating-the-jax-implementation` branch since it contains 
`jax_environment.py`. We only install the `imp-act` package and no dependencies,
```bash
cd imp-act && git checkout 99-updating-the-jax-implementation
poetry install --only-root
```

### 4.Test the installation

To test the installation, run the following command:
```bash
python mava/systems/ppo/anakin/rec_mappo.py env=rware env/scenario=tiny-4ag
```
