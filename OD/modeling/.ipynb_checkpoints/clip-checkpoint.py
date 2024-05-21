import clip

import torch
import torch.nn as nn
import time
import numpy as np
import copy

class SlotAttention(nn.Module):
    def __init__(self, dim=768, iters=3, eps=1e-8, hidden_dim=512, drop_rate=0.4, feature_size=512):
        super().__init__()
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.feature_size = feature_size

        self.to_q = nn.Linear(dim, dim)
        slot_share_qk = False
        if slot_share_qk:
            self.to_k = self.to_q
        else:
            self.to_k = nn.Linear(dim, dim)

        self.to_v = nn.Linear(feature_size, feature_size)

        hidden_dim = max(dim, hidden_dim, feature_size)

        self.gru = nn.GRUCell(feature_size, feature_size)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_size)
        )

        self.norm_slots = nn.LayerNorm(feature_size)
        self.norm_pre_ff = nn.LayerNorm(feature_size)
        self.norm_input = nn.LayerNorm(feature_size)

        self.slot_dropout = nn.Dropout(drop_rate)
        self.input_dropout = nn.Dropout(drop_rate)
    
    def forward(self,cand_feat, pano_feat):

        b, d, device = *pano_feat.shape, pano_feat.device
        # original cand_feat as the initial slot
        slots = cand_feat.clone()
        slots = self.slot_dropout(slots)
        pano_feat = self.norm_input(pano_feat.clone())
        pano_feat = self.input_dropout(pano_feat)
        # (bs, num_ctx, hidden_size)
        k = self.to_k(slots)
        v = self.to_v(slots)
        attn_weights = []
        for t in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots.clone())
            # (bs, num_slots, hidden_size)
            q = self.to_q(pano_feat.clone())
            # (bs, num_slots, num_ctx)
            dots = torch.einsum('id,jd->ijd', k, q) * self.scale

            attn = dots.softmax(dim=1)
            attn_weights.append(attn)   # for visualization
            # (bs, num_slots, feature_size)
            updates = torch.einsum('id,ijd->id', v, attn)
            gru_updates = self.gru(
                updates.reshape(-1, self.feature_size),
                slots_prev.clone().reshape(-1, self.feature_size)
            )
            gru_updates = gru_updates + self.mlp(self.norm_pre_ff(gru_updates))
            slots = gru_updates.clone()
        return slots
        
    
class ClipPredictor(nn.Module):
    def __init__(self, clip_enocder_name,inshape, device, clsnames):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_enocder_name, device)
        self.model.float()
        #freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        # this is only used for inference   
        self.frozen_clip_model = copy.deepcopy(self.model)

        self.visual_enc = self.model.visual
        prompt = 'a photo of a {}'
        print(clsnames)
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(prompt.format(cls)) for cls in clsnames]).to(device)
            self.text_features = self.model.encode_text(text_inputs).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        

        self.projection = nn.Linear(inshape,512)
        self.projection_global = nn.Linear(inshape,512)
    
        self.slot_attention = SlotAttention(
                dim=512,
                iters=3,
                drop_rate=0,
            )   
    
    def forward(self, feat, gfeat=None):

        if feat.shape[-1] > 512:
            feat = self.projection(feat)
        feat  = 0.5* feat + 0.5* self.slot_attention(feat,self.text_features.detach())
        feat = feat/feat.norm(dim=-1,keepdim=True)
        if gfeat is not None:
            
            feat = feat-gfeat
            feat = feat/feat.norm(dim=-1,keepdim=True) 
        scores =  (100.0 * torch.matmul(feat,self.text_features.detach().T))

        # print(scores.min(),scores.max())
        # add for bkg class a score 0
        scores = torch.cat([scores,torch.zeros(scores.shape[0],1,device=scores.device)],1) 
        return scores
                                            
    
    


    


   

    
