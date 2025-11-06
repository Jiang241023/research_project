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
> # Install core stack (match Torch/CUDA to your system)
> uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
> uv pip install -r requirements.txt
>
> # PyG wheels (match Torch 2.8.0 + cu128)
> uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
>   -f https://data.pyg.org/whl/torch-2.8.0+cu128.html --trusted-host data.pyg.org
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