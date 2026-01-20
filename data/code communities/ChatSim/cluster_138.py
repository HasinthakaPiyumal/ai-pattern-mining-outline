# Cluster 138

class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type='CORNER'):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == 'CORNER' or head_type == 'CENTER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor, search: torch.Tensor, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):
        x, aux_dict = self.backbone(z=template, x=search, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, return_last_attn=return_last_attn)
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = enc_opt.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        if self.head_type == 'CORNER':
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map}
            return out
        elif self.head_type == 'CENTER':
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map_ctr, 'size_map': size_map, 'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)

