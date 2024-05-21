import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from ipdb import set_trace as stxx

@BACKBONE_REGISTRY.register()
class ClipRN101(Backbone):
    def __init__(self, cfg, clip_visual):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE 
        self.proj = nn.Linear(512,512)
        self.global_proj = nn.Linear(512,512)
        self.use_proj = cfg.MODEL.USE_PROJ 
        

    def set_backbone_model(self,model):
        self.enc = model
        for name,val in self.enc.named_parameters():
            head = name.split('.')[0]
            if head not in self.unfreeze:
                val.requires_grad = False
            else:
                val.requires_grad = True
        
        self.backbone_unchanged = nn.Sequential(*self.enc.layer3[:19])

    def forward(self, image):

        x = image
    
        batch_num, _, _, _ = x.shape
        
        
        gate_activations = []
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        norm1 = torch.zeros(1, batch_num+1).to(x.device)
        norm2 = torch.zeros(1, batch_num+1).to(x.device)
        flops = torch.zeros(1, batch_num+2).to(x.device)
        
        x = self.enc.layer1(x)
        x  = self.enc.layer2(x)
        x = self.enc.layer3(x)
        return {"res4": x}


    def forward_l12(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)  
       
        return x
     
    def forward_l3(self, x):
        x = self.enc.layer3(x)        
        return {"res4": x}


    def output_shape(self):
        return {"res4": ShapeSpec(channels=1024, stride=16)}
    
#     def forward_res5(self,x):
#     def forward_res5(self, x, norm1, norm2, flops):
#         #detectron used last resnet layer for roi heads
#         x, norm1, norm2, flops = self.enc.layer4((x, norm1, norm2, flops))
# #         x = self.enc.layer4(x)
#         return x



    def forward_res5(self, x, txt_emb, norm1, norm2, flops):
        #detectron used last resnet layer for roi heads
        x, txt_emb, norm1, norm2, flops = self.enc.layer4((x, txt_emb, norm1, norm2, flops))
        return x



    def attention_global_pool(self,input):
        x = input
        x = self.enc.attnpool(x)
        return x
    
    