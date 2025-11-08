# RP-3875
Research Project:  Potential Interaction of Graph Reconstruction Methods and Transformers

# Name
Yueyang Jiang (st186731)

# 1.Python environment setup with uv
> ```bash
> # Install uv (once)
> curl -LsSf https://astral.sh/uv/install.sh | sh
>
> # Inside repo root:
> uv venv
> source .venv/bin/activate
>
> # Install required packages 
> uv pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128
> uv pip install h5py
> uv pip install tqdm
> uv pip install pandas
> uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
>   -f https://data.pyg.org/whl/torch-2.8.0+cu128.html --trusted-host data.pyg.org
> uv pip install rdkit
> uv pip install torchmetrics
> uv pip install ogb
> uv pip install tensorboardX
> uv pip install yacs
> uv pip install opt_einsum
> uv pip install graphgym 
> uv pip install pytorch-lightning
> uv pip install setuptools
> uv pip install matplotlib
> uv pip install scipy
> ```

# 2.The architecture of hybrid_approach
> ```bash
> hybrid_approach/                          
> ├─ config_yaml/                     # the configurations for these approaches.
> ├─ evaluation/                      # where you can implement the evaluation.
> ├─ feature_extraction/              # where you can extract the features.
> ├─ grit_like_and_graphormer_like/   # crucial files for hybrid_approach.
> │  ├─ framework/                    # where you can find components of models such as encoder, head, layer, loader, loss funtion, network, and optimizer.
> │  ├─ main.py                       # where you can run the training loop
> │  ├─ predict.py                    # where you can predict the springback.
> ├─ visualization/                   # where you can implement the visualization.
> ├─ .gitignore
> ├─ LICENSE
> └─ README.md
> ```

# 3.Run the models
> ```bash
>
> ```