import torch
import torch.nn as nn

from models.encoders.single import PerResidueEncoder
from models.encoders.pair import ResiduePairEncoder
from models.encoders.attn import GAEncoder
from models.register import ModelRegister
from models.esm_rinalmo_struct import load_esm, load_rinalmo, segment_cat_pad, cat_pad
from models.lora_tune import LoRAESM, LoRARiNALMo, ESMConfig, RiNALMoConfig
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
)
R = ModelRegister()

@R.register('ipa')
class InvariantPointAttention(nn.Module):
    def __init__(self, 
                 node_feat_dim=640, 
                 rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
                 esm_type='650M',
                 use_lm=False, 
                 fix_lms=True, 
                 pair_feat_dim=64, 
                 num_layers=3, 
                 pooling='mean', 
                 output_dim=1, 
                 representation_layer=33,
                 lora_tune=False,
                 lora_rank=16,
                 lora_alpha=32,
                 **kwargs):
        
        super().__init__()
        self.use_lm = use_lm
        self.fix_lms = fix_lms
        self.proj = 0
        self.representation_layer = representation_layer
        if self.use_lm:
            self.esm, esm_feat_size = load_esm(esm_type)
            self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights)
            if esm_feat_size != rinalmo_feat_size:
                self.proj = 1
                self.project_feat= nn.Linear(esm_feat_size, rinalmo_feat_size)
            self.feat_size = rinalmo_feat_size
            self.proj_cplx= nn.Linear(self.feat_size, node_feat_dim)
            if lora_tune:
                print("Getting Lora!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # copied from LongLoRA
                rinalmo_lora_config = LoraConfig(
                    r=lora_rank,
                    bias="none",
                    lora_alpha=lora_alpha
                )
                esm_lora_config = LoraConfig(
                    r=lora_rank,
                    bias="none",
                    lora_alpha=lora_alpha
                )
                rinalmo_config = RiNALMoConfig()
                esm_config = ESMConfig()
                self.rinalmo = LoRARiNALMo(self.rinalmo, rinalmo_config)
                self.esm = LoRAESM(self.esm, esm_config)
                self.rinalmo = get_peft_model(self.rinalmo, rinalmo_lora_config)
                print("Get RINALMO DONE!!!!!")
                self.esm = get_peft_model(self.esm, esm_lora_config)
                print("Get ESM DONE!!!!!")
            elif fix_lms:
                for p in self.rinalmo.parameters():
                    p.requires_grad_(False)
                for p in self.esm.parameters():
                    p.requires_grad_(False)
        # Encoding
        else:
            self.single_encoder = PerResidueEncoder(
                feat_dim=node_feat_dim,
                max_num_atoms=4,  # N, CA, C, O, CB,
            )
        self.masked_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=node_feat_dim,
            padding_idx=0,
        )
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=pair_feat_dim,
            max_num_atoms=4,  # N, CA, C, O, CB,
        )
        self.pooling = pooling
        self.attn_encoder = GAEncoder(node_feat_dim=node_feat_dim, pair_feat_dim=pair_feat_dim, num_layers=num_layers)
        self.pred_head = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, output_dim)
        )

    def forward(self, input, strategy='separate'):
        mask_residue = input['mask_atoms'][:, :, 1] #CA
        if not self.use_lm:
            x = self.single_encoder(
                aa=input['restype'],
                # phi=batch['phi'], phi_mask=batch['phi_mask'],
                # psi=batch['psi'], psi_mask=batch['psi_mask'],
                # chi=batch['chi'], chi_mask=batch['chi_mask'],
                mask_residue=mask_residue,
            )
        else:
            prot_input = input['prot']
            prot_chains = input['prot_chains']
            prot_mask = input['protein_mask']
            na_input = input['na']
            na_chains = input['na_chains']
            na_mask = input['na_mask']
            with torch.cuda.amp.autocast():
                prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]
                na_embedding = self.rinalmo(na_input)['representation']
                if self.proj:
                    prot_embedding = self.proj_cplx(prot_embedding)
            prot_embedding = prot_embedding.float()
            na_embedding = na_embedding.float()
            # print("Original Embedding:", prot_embedding, na_embedding)
            max_len = input['pos_atoms'].shape[1]
            if 'patch_idx' in input:
                patch_idx = input['patch_idx']
            else:
                patch_idx = None
            # Adjust the embeddings from LMs for CFormer
            if strategy == 'separate':
                # input shape [N', L], where N' is flexible in every batch
                out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx)
                assert out_embedding.shape[0] == input['size']
            else:
                out_embedding, masks = cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx)
                assert out_embedding.shape[0] == input['size']
            x = self.proj_cplx(out_embedding)

        aa=input['restype']
        res_nb=input['res_nb']
        chain_nb=input['chain_nb']
        pos_atoms=input['pos_atoms']
        mask_atoms=input['mask_atoms']
        
        if self.pooling == 'token':
            mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
            cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
            prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
            rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)
            
            out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
            key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
            
            cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
            prot_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_PROT_IDX
            rna_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_RNA_IDX
            aa = torch.cat([cplx_type, prot_type, rna_type, aa], dim=1)
            
            res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
            res_nb_prot = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 1
            res_nb_rna = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 2
            
            res_nb = torch.cat([res_nb_cplx, res_nb_prot, res_nb_rna, res_nb], dim=1)
            super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
            chain_nb = torch.cat([super_chain_id, super_chain_id, super_chain_id, chain_nb], dim=1)
            
            center_cplx = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2], 3), device=out_embedding.device, dtype=pos_atoms.dtype)
            center_prot = ((pos_atoms * input['identifier'][:, :, None, None] * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / (input['identifier'][:, :, None] * mask_atoms).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            center_rna = ((pos_atoms * (1-input['identifier'][:, :, None, None]) * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((1-input['identifier'][:, :, None]) * mask_atoms).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            pos_atoms = torch.cat([center_cplx, center_prot, center_rna, pos_atoms], dim=1)
            mask_atom = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2]), device=out_embedding.device, dtype=mask_atoms.dtype)
            mask_atom[:,:,0] = 1
            mask_atoms = torch.cat([mask_atom, mask_atom, mask_atom, mask_atoms], dim=1)


        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
        )
        
        x = self.attn_encoder(
            pos_atoms=input['pos_atoms'],
            res_feat=x, pair_feat=z,
            mask=mask_residue
        )
        if self.pooling == 'token':
            complex_embedding = x[:, 0, :]
        else:
            complex_embedding = (x * input['seq_mask'].unsqueeze(-1)).sum(dim=1)
            if self.pooling == 'mean':
                # Prot_mask: [N, L]
                complex_mask_sum = input['seq_mask'].sum(dim=1, keepdim=True)
                complex_embedding = complex_embedding / (complex_mask_sum + 1e-10)

        output = self.pred_head(complex_embedding)
        output = output.squeeze(1)
        
        return output