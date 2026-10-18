import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        
        self.has_weights_merged = False
        if lora_rank > 0:
            
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_scaling = lora_alpha / lora_rank

            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
             
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, self.bias)
        if self.is_lora() and not self.has_weights_merged:
            dropped_input = self.lora_dropout(input)
            h = dropped_input @ self.lora_A.T
            lora_output = h @ self.lora_B.T
            result = result + lora_output * self.lora_scaling

        return result

        

    def train(self, mode: bool = True) -> "LoRALinear":
        super().train(mode)

        if self.is_lora():
            if mode and self.has_weights_merged:
            # Switching into training: unmerge
                with torch.no_grad():
                    self.weight -= (
                    self.lora_B @ self.lora_A
                ) * self.lora_scaling

                self.has_weights_merged = False

            elif not mode and not self.has_weights_merged:
            # Switching into evaluation: merge
                with torch.no_grad():
                    self.weight += (
                    self.lora_B @ self.lora_A
                ) * self.lora_scaling

                self.has_weights_merged = True

        return self

    def eval(self) -> "LoRALinear":
        return self.train(False)
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:

    for name, param in model.named_parameters():
        param.requires_grad = ("lora_" in name)

    return model
