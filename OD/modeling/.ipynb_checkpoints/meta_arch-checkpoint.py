from ast import mod
import math
import numpy as np
import cv2
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from typing import Dict,List,Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
# from .regularization import *

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(feat, input_img_tensor, scale_factor=60, size=(224, 224)):
    """

    :param feat: 3D ,4D
    :param input_img_tensor:
    :param size:
    :return:
    """
    input_img_tensor = input_img_tensor[:, :size[0], :size[1]]
    feat = torch.nn.functional.interpolate(feat, scale_factor=scale_factor, mode='bilinear')
    feat = feat.reshape(size).cuda().data.cpu().numpy()
    feat = (feat - feat.min()) / (
            feat.max() - feat.min())
    image_feat = input_img_tensor.permute(1, 2, 0).data.cpu().numpy()
    image_feat = (image_feat - image_feat.min()) / (
            image_feat.max() - image_feat.min())
    vis = show_cam_on_image(image_feat, feat)
    vis = np.uint8(255 * vis)
    return vis


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)

        # txt 
        domain_text = {'day': 'an image taken during the day'}
        with open('prunedprompts2.txt','r') as f:
            for ind,l in enumerate(f):
                domain_text.update({str(ind):l.strip()})
        self.offsets = nn.Parameter(torch.zeros(len(domain_text)-1,1024,14,14)) #skip day

        import clip
        self.domain_tk = dict([(k,clip.tokenize(t)) for k,t in domain_text.items()])
        self.apply_aug = cfg.AUG_PROB
        
        day_text_embed_list = []
        for i,val in enumerate(self.domain_tk.items()):
            name , dtk = val
            if name == 'day':
                continue
            with torch.no_grad():
                
                day_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(self.domain_tk['day'].cuda()) #day
                day_text_embed = day_text_embed/day_text_embed.norm(dim=-1,keepdim=True)
                day_text_embed_list.append(day_text_embed)
        self.day_text_embeds = torch.cat(day_text_embed_list,0).cuda()    
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean=[0.48145466, 0.4578275, 0.40821073]
        std=[0.26862954, 0.26130258, 0.27577711] 
  

        clip_images = [ T.functional.normalize(ci.flip(0)/255, mean,std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i  for i in clip_images])
        return clip_images


    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]#batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        features = self.backbone(images.tensor)
        
        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)                
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self,N):
        import colorsys
        '''
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        '''
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

            
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
         
       
        ###############save feat vis###################
        feat_vis = False
        if feat_vis:
            scale_size = 16
            out_dir_explain = os.path.join('./output', 'featmap_vis')
            explain_rpn_feat_i, _ = torch.max(features['res4'][0], 0)
            explain_rpn_feat_i = explain_rpn_feat_i.unsqueeze(0).unsqueeze(0)

            size = (explain_rpn_feat_i.shape[-2] * scale_size , explain_rpn_feat_i.shape[-1] *scale_size)
            visual = generate_visualization(explain_rpn_feat_i, images.tensor[0], scale_factor=scale_size, size=size)

            name_sp_list = batched_inputs[0]['file_name'].split('/')[-1].rsplit('.', 1)
            save_file_name = name_sp_list[0] + '.' + name_sp_list[1]
            explain_out_file = os.path.join(out_dir_explain, save_file_name)
            cv2.imwrite(explain_out_file, visual)        
        
        ###############save feat vis ###################        
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                logits,proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
             
            try:
                results, _ = self.roi_heads(images,self.day_text_embeds, features, proposals, None, None, self.backbone)
            except:
                results, _ = self.roi_heads(images,self.day_text_embeds, features, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)


            return allresults
        else:
            return results


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneTrainable(ClipRCNNWithClipBackbone):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]#batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)                
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        _, detector_losses = self.roi_heads(images, self.day_text_embeds, features, proposals, gt_instances, None, self.backbone)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    

