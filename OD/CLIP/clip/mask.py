import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

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
        b, d = pano_feat.shape
        # original cand_feat as the initial slot
        slots = cand_feat.clone()
        slots = self.slot_dropout(slots)

        pano_feat = self.norm_input(pano_feat.clone())
        pano_feat = self.input_dropout(pano_feat)

        # (bs, num_ctx, hidden_size)
        k = self.to_k(pano_feat)
        v = self.to_v(pano_feat)
        attn_weights = []

        for t in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots.clone())

            # (bs, num_slots, hidden_size)
            q = self.to_q(slots.clone())

            # (bs, num_slots, num_ctx)
            dots = torch.einsum('id,jd->ij', q, k) * self.scale

            attn = dots.softmax(dim=1)

            attn_weights.append(attn)   # for visualization

            # (bs, num_slots, feature_size)
            updates = torch.einsum('jd,ij->id', v, attn)

            gru_updates = self.gru(
                updates.reshape(-1, self.feature_size),
                slots_prev.clone().reshape(-1, self.feature_size)
            )
            gru_updates = gru_updates + self.mlp(self.norm_pre_ff(gru_updates))

            slots = gru_updates.clone()

        return slots # , np.stack([a.cpu().detach().numpy() for a in attn_weights], 0)


class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''
    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
    
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor+eps)-torch.log(
                                          1-uniform_samples_tensor+eps)
        return gumble_samples_tensor
    
    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits
    
    def forward(self, logits):
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard



class Mask_s(nn.Module):
    '''
        Attention Mask spatial.
    '''
    def __init__(self, h, w, planes, block_w, block_h, eps=0.66667,
                 bias=-1, **kwargs):
        super(Mask_s, self).__init__()
        # Parameter
        self.width, self.height, self.channel = w, h, planes
        self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        self.eleNum_s = torch.Tensor([self.mask_h*self.mask_w])
        # spatial attention
        self.atten_s = nn.Conv2d(planes, 1, kernel_size=3, stride=1, bias=bias>=0, padding=1)
        if bias>=0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x):

        batch, channel, height, width = x.size() # torch.Size([256, 64, 56, 56])
        # Pooling
        input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w)) # torch.Size([256, 64, 7, 7])
        # spatial attention
        s_in = self.atten_s(input_ds) # [N, 1, h, w]
        
        # spatial gate
        mask_s = self.gate_s(s_in) # [N, 1, h, w]
        # norm
        norm = self.norm(mask_s)
        norm_t = self.eleNum_s.to(x.device)
        return mask_s, norm, norm_t
    
    def get_flops(self):
        flops = self.mask_h * self.mask_w * self.channel * 9
        return flops


class Mask_c(nn.Module):
    '''
        Attention Mask.
    '''
    def __init__(self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, **kwargs):
        super(Mask_c, self).__init__()
        # Parameter
        self.bottleneck = 512 # inplanes // fc_reduction 
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c_fc1 = nn.Conv2d(inplanes, 512, kernel_size=1)
        self.slot_attention = SlotAttention(
                dim=512,
                iters=3,
                drop_rate=0,
            )
        
        self.atten_c_bn = nn.BatchNorm2d(self.bottleneck)
        self.atten_c_act = nn.ReLU(inplace=True)
        self.atten_c_conv = nn.Conv2d(self.bottleneck, outplanes, kernel_size=1, stride=1, bias=bias>=0)

        if bias>=0:
            nn.init.constant_(self.atten_c_conv.bias, bias)
        # Gate
        self.gate_c = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x, txt_emb):
        batch, channel, _, _ = x.size()
        context = self.avg_pool(x) # [N, C, 1, 1] 
        context = self.atten_c_fc1(context)
        # transform
        c_in = context+self.slot_attention(context.squeeze(-1).squeeze(-1),txt_emb.detach()).unsqueeze(-1).unsqueeze(-1)
        c_in = self.atten_c_bn(c_in)
        c_in = self.atten_c_act(c_in)        
        c_in = self.atten_c_conv(c_in)
        
        # channel gate
        mask_c = self.gate_c(c_in) # [N, C_out, 1, 1]
        # norm
        norm = self.norm(mask_c)
        norm_t = self.eleNum_c.to(x.device)
        return mask_c, norm, norm_t
    
    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops
