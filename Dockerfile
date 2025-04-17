# Use a CUDA runtime image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget curl bzip2 vim git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Set PATH so conda is accessible everywhere
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Clone repositories
RUN git clone https://github.com/AI-for-Infrastructure-Management/imp-act-Mava.git && \
    cd imp-act-Mava && git checkout imp_act_adaption && \
    git clone https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git && \
    cd imp-act-JaxMARL && git checkout qlearning_rnn && cd .. && \
    git clone https://github.com/AI-for-Infrastructure-Management/imp-act.git && \
    cd imp-act && git checkout 99-updating-the-jax-implementation && cd ..

# Create Conda environment (impact-mava-env)
RUN conda env create -f imp-act-Mava/conda_environment.yaml

# Set conda environment path (for clarity)
ENV CONDA_DEFAULT_ENV=impact-mava-env
ENV PATH=$CONDA_DIR/envs/impact-mava-env/bin:$PATH

# Install Python Packages

# 1. Modify Jax dependency for CUDA
RUN sed -i 's/jax==\(.*\)/jax[cuda12]==\1/' imp-act-Mava/requirements/requirements.txt

# 2. Install Mava
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate impact-mava-env && \
    cd imp-act-Mava && \
    pip install --no-cache-dir -e '.[algs]'"
# Undo the Jax dependency modification
RUN sed -i 's/jax\[cuda12\]==\(.*\)/jax==\1/' imp-act-Mava/pyproject.toml

# 3. Install JaxMARL
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate impact-mava-env && \
    cd imp-act-Mava/imp-act-JaxMARL && \
    pip install --no-deps -e '.[algs]'"

# 4. Install imp-act
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate impact-mava-env && \
    cd imp-act-Mava/imp-act && \
    poetry install --only-root"

# Clean up
RUN rm -rf ~/.cache/pip

# Persist environment variables
RUN printenv > /etc/environment