from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


class GSR(nn.Module):

    def __init__(self, backbone, transformer, num_nouns, num_verbs, num_roles, verb_roles, role_mask, aux_loss=False, use_verb=True):
        super().__init__()

        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        if use_verb:
            self.verb_emb_layer = nn.Embedding(num_verbs,
                                               hidden_dim)
        self.use_verb = use_verb
        self.role_emb_layer = nn.Embedding(num_roles + 1,
                                           hidden_dim, padding_idx=num_roles)

        self.obj_class_embed = nn.Linear(hidden_dim, num_nouns)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.register_buffer('verb_roles', verb_roles)
        self.register_buffer('role_mask', role_mask)
        self.pool = nn.Sequential(nn.MaxPool2d(3, 3), nn.Conv2d(2048, 2048, 1))

    def forward(self, samples: NestedTensor, gt_verb):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        assert mask is not None

        if self.use_verb:
            dec_verb_emb = self.verb_emb_layer(gt_verb).unsqueeze(1) # [n, 1, c]

        verb_roles_embedding = self.verb_roles[gt_verb] # [n,6]
        verb_role_mask = self.role_mask[gt_verb].bool() #[n,6]
        pad_role_mask = torch.zeros((verb_role_mask.size(0),1)).cuda().bool()
        verb_role_mask = torch.cat([pad_role_mask,verb_role_mask],dim=1)

        dec_role_embed = self.role_emb_layer(verb_roles_embedding) #[n,6,c]

        if self.use_verb:
            dec_input = torch.cat([dec_verb_emb, dec_role_embed], dim=1).permute(1,0,2)
        else:
            dec_input = dec_role_embed.permute(1,0,2)

        hs, memory = self.transformer(self.input_proj(src), mask, dec_input, pos[-1], verb_role_mask)

        if self.use_verb:
            dec_role_feat = hs[:, :, 1:, :]
        else:
            dec_role_feat = hs
        outputs_obj_class = self.obj_class_embed(dec_role_feat).permute(0, 1, 3, 2).contiguous() #[layers, n, num_roles, num_nouns]
        outputs_coord = self.bbox_embed(dec_role_feat).sigmoid()

        out = {'pred_obj_logits': outputs_obj_class[-1],
                'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class,
                                                    outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_boxes': c}
        #         for a, b, c in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
        #                               outputs_coord[:-1])]
        return [{'pred_obj_logits': a, 'pred_boxes': c}
                for a, c in zip(outputs_obj_class[:-1], outputs_coord[:-1])]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




