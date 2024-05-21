from typing import Dict, List, Optional, Tuple
import torch

from detectron2.layers import cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import  FastRCNNOutputLayers
from .clip import ClipPredictor

class ClipFastRCNNOutputLayers(FastRCNNOutputLayers):

    def __init__(self,cfg, input_shape, clsnames) -> None:
        super().__init__(cfg, input_shape)
        self.cls_score = ClipPredictor(cfg.MODEL.CLIP_IMAGE_ENCODER_NAME, input_shape.channels, cfg.MODEL.DEVICE,clsnames)
    def forward(self,x,gfeat=None):

        if isinstance(x,list):
            scores = self.cls_score(x[0],gfeat)
            proposal_deltas = self.bbox_pred(x[1])
        else: 
            scores = self.cls_score(x,gfeat)
            proposal_deltas = self.bbox_pred(x)

        return scores, proposal_deltas  



