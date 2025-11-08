# RP-3875
Research Project:  Potential Interaction of Graph Reconstruction Methods and Transformers

> **Python environment setup with uv**
>
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

> **The architecture of hybrid approaches**
> ```bash
>   hybrid_approach/  
> ├─ config_yaml/  
> │  ├─ ddacs-node-regression.yaml  
> │  └─ ddacs-node-regression-graphormerlike.yaml  
> ├─ feature_extraction/  
> │  └─ dataset_feature_extraction.py  
> ├─ grit_like_and_graphormer_like/  
> │  ├─ framework/                # model/layer/loader registers  
> │  ├─ main.py                   # train  
> │  ├─ predict.py                # predict  
> │  └─ visualization/
> │     └─ visualization.py         
> ├─ Evaluation/  
> │  └─ evaluation.py  
> ├─ README.md  
> └─ requirements.txt  
> ```