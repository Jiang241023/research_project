import torch, platform

print("Torch:", torch.__version__)                  # e.g. 2.4.0+cu121 or 2.4.0
print("CUDA (build):", torch.version.cuda)          # e.g. '12.1' or None (CPU build)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    try:
        print("cuDNN:", torch.backends.cudnn.version())
    except Exception as e:
        print("cuDNN: N/A", e)

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cu128.html --trusted-host data.pyg.org