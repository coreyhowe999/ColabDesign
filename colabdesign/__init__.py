import os,jax
# disable triton_gemm for jax versions > 0.3
if int(jax.__version__.split(".")[1]) > 3:
  os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign.shared.utils import clear_mem, add_cyclic_offset
from colabdesign.af.model import mk_af_model
from colabdesign.tr.model import mk_tr_model
from colabdesign.mpnn.model import mk_mpnn_model

# backward compatability
mk_design_model = mk_afdesign_model = mk_af_model
mk_trdesign_model = mk_tr_model
