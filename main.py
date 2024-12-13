import math
import torch
import torch.nn as nn
from labml_helpers.module import Module
from typing import List
from labml import monit
from labml_nn.neox.model import LayerGenerator
from labml_nn.neox.utils import get_tokens, print_tokens
from labml_nn.neox.utils.cache import get_cache
from typing import Optional, List
LAYERS = None
PROMPT = 'Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German'
def infer(model: nn.Module, ids: List[int], device: torch.device):
  with torch.no_grad():
    x = torch.tensor(ids)[None, :].to(device)
    x = model(x)
  return x[0].max(dim=-1)[1].tolist()


def generate():
    cache = get_cache()
    cache.set('use_cache', True)

    device = torch.device('cuda:0')

    layers = list(LayerGenerator(is_clone_layers=True,
    filter_layers = LAYERS,
    dtype = torch.float16,

    device = device,).load())

    model = nn.Sequential(*layers)

    ids = get_tokens(PROMPT)
    #


    cache.set('state_ids', (None, 1))

    with monit.section('Infer'):

        next_token = infer(model, ids, device)[-1]
    #

    ids += [next_token]
    #

    for i in range(1, 100):



       cache.set('state_ids', (i, i + 1))

       with monit.section('Infer'):

          next_token = infer(model, [next_token], device)[-1]
    #

          ids += [next_token]
    #


          print_tokens(ids, [ids])
    #


if __name__ == '__main__':



    generate()