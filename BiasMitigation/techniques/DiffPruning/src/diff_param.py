import torch
from  torch import nn
from torch.nn.parameter import Parameter
#import torch.nn.utils.parametrize as parametrize

from typing import Union

from techniques.DiffPruning.src.utils import concrete_stretched


class DiffWeight(nn.Module):
    """
    Implementation of diff pruning weights using pytorch parametrizations
    https://pytorch.org/tutorials/intermediate/parametrizations.html
    """

    def __init__(
        self,
        weight: torch.Tensor,
        alpha_init: Union[float, int],
        concrete_lower: Union[float, int],
        concrete_upper: Union[float, int],
        structured: bool
    ):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured
        
        weight.requires_grad = False
        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))
        
        if structured:
            self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))
              
    def forward(self, X):
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.z * (self.finetune - X)
    
    @property
    def z(self) -> Parameter:
        z = self.dist(self.alpha)
        if self.structured:
            z *= self.dist(self.alpha_group)
        return z
    
    @property
    def alpha_weights(self) -> list:
        alpha = [self.alpha]
        if self.structured:
            alpha.append(self.alpha_group)
        return alpha

    def dist(self, x) -> torch.Tensor:
        return concrete_stretched(
            x,
            l=self.concrete_lower,
            r=self.concrete_upper,
            deterministic=(not self.training)
        )
            
            
class DiffWeightFixmask(nn.Module):
    
    def __init__(self, pre_trained: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("pre_trained", Parameter(pre_trained, requires_grad=False))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
    
    def forward(self, X):
        return self.pre_trained + self.mask * X    
