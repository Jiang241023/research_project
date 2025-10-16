from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """

    # Argument group for each Positional Encoding class.
    cfg.posenc_RWSE = CN()
    cfg.posenc_RFF = CN()
    cfg.posenc_PPRAnchors = CN()

    # Common arguments.
    for name in ['posenc_RWSE', 'posenc_RFF', 'posenc_PPRAnchors']:
        pecfg = getattr(cfg, name)
        # Use extended positional encodings
        pecfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    for name in ['posenc_RWSE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''





