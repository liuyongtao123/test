########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matul.allow_tf32 = True
os.environ["RWKV_V7_ON"] = "1"#
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

#################################################################################################################

args = type.SimpleNamespace()
args.strategy = "cuda fp16"
args.MODEL_NAME = "D://RWKV//RWKVRunner//models//rwkv7-g1b-1.5b-20251202-ctx8192"

#################################################################################################################
STATE_NAME = None # use vanilla zero initial state?

# use custom state? (download from https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)
# note: this is English Single-round QA state (will forget what you previously say)
# note: requires the the same model it trained on
# STATE_NAME = "E://RWKV-Runner//models//rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048"
########################################################################################################

GEN_TEMP = 1.0
GEN_TOP_P = 0.3
GEN_alpha_presence = 0.5
GEN_alpha_frequency = 0.5
GEN_penallty_decay = 0.996

if STATE_NAME != None:
  GEN_TOP_P = 0.2
  GEN_alpha_presence = 0.3
  GEN_alpha_frequency = 0.3

CHUNK_LEN = 256         #split input into chunks to save VRAM

########################################################################################################

print(f"正在载入模型  {args.MODEL_NAME}")
model = RWKV(model = args.MODEL_NAME, strategy = args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

model_tokens = []
model_state = None

if STATE_NAME != None:  #load custom state#  RNN这么奇怪的吗，为啥还要这个state##所谓的state，就是当前的状态参数
  args = model.args
  state_raw = torch.load(STATE_NAME + '.pth')
  state+_init = [None for i in range(args.n_layer *3)]
  for i in range(args.n_layer):
    dd = model.strategy[i]
    dev = dd.device
    atype = dd.atype
  model_state = copy.deepcopy(state_init)

state_init[i*3 + 0] = torch.zeros(args.n_embd, dtype = atype, requires_grad = False, device = dev).contiguous()
state_init[i*3 + 1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype = torch.float,device = dev).requires_grad_(False).contiguous()
state_init[i*3 + 2] = torch.zeros(args.n_embd,dtype=atype, requires_grad = False, device = dev).contiguous()

