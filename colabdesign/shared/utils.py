import random
import jax
import numpy as np
import jax.numpy as jnp
import sys, gc


def add_cyclic_offset(self, offset_type=2):
  '''add cyclic offset to connect N and C term'''
  def cyclic_offset(L):
    i = np.arange(L)
    ij = np.stack([i,i+L],-1)
    offset = i[:,None] - i[None,:]
    c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
    if offset_type == 1:
      c_offset = c_offset
    elif offset_type >= 2:
      a = c_offset < np.abs(offset)
      c_offset[a] = -c_offset[a]
    if offset_type == 3:
      idx = np.abs(c_offset) > 2
      c_offset[idx] = (32 * c_offset[idx] )/  abs(c_offset[idx])
    return c_offset * np.sign(offset)
  idx = self._inputs["residue_index"]
  offset = np.array(idx[:,None] - idx[None,:])

  if self.protocol == "binder":
    c_offset = cyclic_offset(self._binder_len)
    offset[self._target_len:,self._target_len:] = c_offset

  if self.protocol in ["fixbb","partial","hallucination"]:
    Ln = 0
    for L in self._lengths:
      offset[Ln:Ln+L,Ln:Ln+L] = cyclic_offset(L)
      Ln += L
  self._inputs["offset"] = offset

def clear_mem():
  # clear vram (GPU)
  backend = jax.lib.xla_bridge.get_backend()
  if hasattr(backend,'live_buffers'):
    for buf in backend.live_buffers():
      buf.delete()

  # TODO: clear ram (CPU)
  gc.collect()
  
def update_dict(D, *args, **kwargs):
  '''robust function for updating dictionary'''
  def set_dict(d, x, override=False):
    for k,v in x.items():
      if v is not None:
        if k in d:
          if isinstance(v, dict):
            set_dict(d[k], x[k], override=override)
          elif override or d[k] is None:
            d[k] = v
          elif isinstance(d[k],(np.ndarray,jnp.ndarray)):
            d[k] = np.asarray(v)
          elif isinstance(d[k], dict):
            d[k] = jax.tree_map(lambda x: type(x)(v), d[k])
          else:
            d[k] = type(d[k])(v)
        else:
          print(f"ERROR: '{k}' not found in {list(d.keys())}")  
  override = kwargs.pop("override", False)
  while len(args) > 0 and isinstance(args[0],str):
    D,args = D[args[0]],args[1:]
  for a in args:
    if isinstance(a, dict): set_dict(D, a, override=override)
  set_dict(D, kwargs, override=override)

def copy_dict(x):
  '''deepcopy dictionary'''
  return jax.tree_map(lambda y:y, x)

def to_float(x):
  '''convert to float'''
  if hasattr(x,"tolist"): x = x.tolist()
  if isinstance(x,dict): x = {k:to_float(y) for k,y in x.items()}
  elif hasattr(x,"__iter__"): x = [to_float(y) for y in x]
  else: x = float(x)
  return x

def dict_to_str(x, filt=None, keys=None, ok=None, print_str=None, f=2):
  '''convert dictionary to string for print out'''  
  if keys is None: keys = []
  if filt is None: filt = {}
  if print_str is None: print_str = ""
  if ok is None: ok = []

  # gather keys
  for k in x.keys():
    if k not in keys:
      keys.append(k)

  for k in keys:
    if k in x and (filt.get(k,True) or k in ok):
      v = x[k]
      if isinstance(v,float):
        if int(v) == v:
          print_str += f" {k} {int(v)}"
        else:
          print_str += f" {k} {v:.{f}f}"
      else:
        print_str += f" {k} {v}"
  return print_str

class Key():
  '''random key generator'''
  def __init__(self, key=None, seed=None):
    if key is None:
      self.seed = random.randint(0,2147483647) if seed is None else seed
      self.key = jax.random.PRNGKey(self.seed) 
    else:
      self.key = key
  def get(self, num=1):
    if num > 1:
      self.key, *sub_keys = jax.random.split(self.key, num=(num+1))
      return sub_keys
    else:
      self.key, sub_key = jax.random.split(self.key)
      return sub_key

def softmax(x, axis=-1):
  x = x - x.max(axis,keepdims=True)
  x = np.exp(x)
  return x / x.sum(axis,keepdims=True)

def categorical(p):
  return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)

def to_list(xs):
  if not isinstance(xs,list): xs = [xs]
  return [x for x in xs if x is not None]

def copy_missing(a,b):
  for i,v in a.items():
    if i not in b:
      b[i] = v
    elif isinstance(v,dict):
      copy_missing(v,b[i])
