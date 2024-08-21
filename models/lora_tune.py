from transformers import PreTrainedModel, BertConfig, PretrainedConfig, BertForMaskedLM
from typing import List
class ESMConfig(PretrainedConfig):
    model_type = "esm2"
    def __init__(
        self,
        pooling='mean',
        output_dim=1,
        fix_lms=True,
        lora_tune=True,
        **kwargs,
    ):
        self.pooling = pooling
        self.output_dim = output_dim
        self.fix_lms = fix_lms
        self.lora_tune = lora_tune
        super().__init__(**kwargs)

class RiNALMoConfig(PretrainedConfig):
    model_type = "rinalmo_flash"
    def __init__(
        self,
        pooling='mean',
        output_dim=1,
        fix_lms=True,
        lora_tune=True,
        **kwargs,
    ):
        self.pooling = pooling
        self.output_dim = output_dim
        self.fix_lms = fix_lms
        self.lora_tune = lora_tune
        super().__init__(**kwargs)


class LoRAESM(PreTrainedModel):
    config_class = ESMConfig
    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        return self.model(tokens, repr_layers, need_head_weights, return_contacts)
    
class LoRARiNALMo(PreTrainedModel):
    config_class = RiNALMoConfig
    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
    def forward(self, tokens, need_attn_weights=False):
        return self.model(tokens, need_attn_weights)