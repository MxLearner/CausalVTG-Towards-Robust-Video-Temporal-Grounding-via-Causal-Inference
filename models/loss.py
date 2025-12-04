# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.
# Modified by Qiyi Wang

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import LOSSES, Parameter, build_loss


@LOSSES.register()
class SampledNCELoss(nn.Module):

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 direction=('row', 'col'),
                 loss_weight=1.0):
        super(SampledNCELoss, self).__init__()

        scale = torch.Tensor([math.log(1 / temperature)])

        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction, ) if isinstance(direction, str) else direction
        self.loss_weight = loss_weight

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}'
                .format(self.temperature, self.max_scale, self.learnable, self.direction,
                        self.loss_weight))

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        i_sim = i_sim + torch.where(loss_msk > 0, .0, float('-inf'))

        loss = 0

        if 'row' in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip]
            loss = loss - i_met.sum() / i_met.size(0)

        if 'col' in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        loss = loss * self.loss_weight
        return loss


@LOSSES.register()
class Giou1DLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        """
        一维 GIoU 损失模块
        Args:
            loss_weight (float): 损失权重，用于多任务损失融合
        """
        super(Giou1DLoss, self).__init__()
        self.loss_weight = loss_weight

    def extra_repr(self):
        return f'loss_weight={self.loss_weight}'

    def giou_1d(self, pred, target):
        """
        pred, target: Tensor of shape [M, 2]
        返回: Tensor of shape [M]，每个样本的 GIoU 损失
        """
        pred_left = torch.min(pred[:, 0], pred[:, 1])
        pred_right = torch.max(pred[:, 0], pred[:, 1])
        tgt_left = torch.min(target[:, 0], target[:, 1])
        tgt_right = torch.max(target[:, 0], target[:, 1])

        inter_left = torch.max(pred_left, tgt_left)
        inter_right = torch.min(pred_right, tgt_right)
        inter = (inter_right - inter_left).clamp(min=0)

        union = (pred_right - pred_left) + (tgt_right - tgt_left) - inter + 1e-6
        iou = inter / union

        enclosing = (torch.max(pred_right, tgt_right) - torch.min(pred_left, tgt_left)).clamp(min=1e-6)
        giou = iou - (enclosing - union) / enclosing

        return 1 - giou  # loss

    def forward(self, pred, target, weight=None, avg_factor=1):
        """
        pred: [B, N, 2]
        target: [B, N, 2]
        weight: [B, N, 2] or [B, N]，用于筛选有效样本
        """
        if weight is not None:
            if weight.shape[-1] == 2:
                weight = weight.all(dim=-1)  # [B, N]
            weight = weight.view(-1)  # [B*N]

        pred = pred.view(-1, 2)
        target = target.view(-1, 2)

        if weight is not None:
            pred = pred[weight]
            target = target[weight]

        loss = self.giou_1d(pred, target)
        return loss.mean() * self.loss_weight

@LOSSES.register()
class BundleLoss(nn.Module):

    def __init__(self,
                 sample_radius=1.5,
                 loss_cls=None,
                 loss_reg=None,
                 loss_sal=None,
                 loss_video_cal=None,
                 loss_layer_cal=None,
                 loss_neg = None):
        super(BundleLoss, self).__init__()

        self._loss_cls = build_loss(loss_cls)
        self._loss_reg = build_loss(loss_reg)
        self._loss_sal = build_loss(loss_sal)
        self._loss_video_cal = build_loss(loss_video_cal)
        self._loss_layer_cal = build_loss(loss_layer_cal)
        self._loss_neg =build_loss(loss_neg)

        self.sample_radius = sample_radius

    def get_target_single(self, point, gt_bnd, gt_cls):
        '''
        eg:
        point = [[0,0,2,1],[1,0,2,1],[2,0,2,1],...,
                      [0,2,4,2],[2,2,4,2],[4,2,4,2],...,
                      [0,4,8,4],[4,4,8,4],[8,4,8,4],...,
                      [0,8,inf,8],[8,8,inf,8],[16,8,inf,8]...]
        每行格式 [位置, 最小距离, 最大距离, 缩放因子]
        gt_bnd = [[0,5],[7,9]]
        gt_cls = [[1],[1]]
        '''
        num_pts, num_gts = point.size(0), gt_bnd.size(0)

        lens = gt_bnd[:, 1] - gt_bnd[:, 0]   # [nums_gts]
        lens = lens[None, :].repeat(num_pts, 1) # [num_pts, num_gts]

        gt_seg = gt_bnd[None].expand(num_pts, num_gts, 2)  # [num_pts, num_gts, 2]
        s = point[:, 0, None] - gt_seg[:, :, 0]  # [num_pts, num_gts] 
        e = gt_seg[:, :, 1] - point[:, 0, None]  # [num_pts, num_gts] 
        r_tgt = torch.stack((s, e), dim=-1) # [num_pts, num_gts,2]

        if self.sample_radius > 0:
            center = (gt_seg[:, :, 0] + gt_seg[:, :, 1]) / 2 # [num_pts, num_gts]
            t_mins = center - point[:, 3, None] * self.sample_radius  # [num_pts, num_gts]
            t_maxs = center + point[:, 3, None] * self.sample_radius # [num_pts, num_gts]
            dist_s = point[:, 0, None] - torch.maximum(t_mins, gt_seg[:, :, 0])  
            dist_e = torch.minimum(t_maxs, gt_seg[:, :, 1]) - point[:, 0, None]
            center = torch.stack((dist_s, dist_e), dim=-1)
            cls_msk = center.min(-1)[0] >= 0
        else:
            cls_msk = r_tgt.min(-1)[0] >= 0

        reg_dist = r_tgt.max(-1)[0]
        reg_msk = torch.logical_and((reg_dist >= point[:, 1, None]),
                                    (reg_dist <= point[:, 2, None]))
        # 上面的代码应该是构建正样本，即那些选点在gt_bnd内的点
        lens.masked_fill_(cls_msk == 0, float('inf'))
        lens.masked_fill_(reg_msk == 0, float('inf'))
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)),
                                         (lens < float('inf'))).to(r_tgt.dtype)

        label = F.one_hot(gt_cls[:, 0], 2).to(r_tgt.dtype)
        c_tgt = torch.matmul(min_len_mask, label).clamp(min=0.0, max=1.0)[:, 1]
        r_tgt = r_tgt[range(num_pts), min_len_inds] / point[:, 3, None]

        # if True:
        #     # c_tgt 或 r_tgt 为 0 时，print
        #     if (reg_msk == 0).all() or (cls_msk == 0).all():
        #         print(f'boundary: {data['boundary'][i]}')    

        return c_tgt, r_tgt

    def get_target(self, data):
        cls_tgt, reg_tgt = [], []

        for i in range(data['boundary'].size(0)):
            gt_bnd = data['boundary'][i] * data['fps'][i]
            gt_cls = gt_bnd.new_ones(gt_bnd.size(0), 1).long()

            c_tgt, r_tgt = self.get_target_single(data['point'], gt_bnd, gt_cls)



            cls_tgt.append(c_tgt)
            reg_tgt.append(r_tgt)

        cls_tgt = torch.stack(cls_tgt)
        reg_tgt = torch.stack(reg_tgt)

        return cls_tgt, reg_tgt

    def loss_cls(self, data, output, cls_tgt):
        src = data['out_class'].squeeze(-1)
        msk = torch.cat(data['pymid_msk'], dim=1)

        loss_cls = self._loss_cls(src, cls_tgt, weight=msk, avg_factor=msk.sum())

        output['loss_cls'] = loss_cls
        return output

    def loss_reg(self, data, output, cls_tgt, reg_tgt):
        src = data['out_coord']
        msk = cls_tgt.unsqueeze(2).repeat(1, 1, 2).bool()

        loss_reg = self._loss_reg(src, reg_tgt, weight=msk, avg_factor=msk.sum())

        output['loss_reg'] = loss_reg
        return output

    def loss_sal(self, data, output):
        video_emb = data['video_emb']
        query_emb = data['query_emb']
        video_msk = data['video_msk']

        saliency = data['saliency']
        pos_clip = data['pos_clip'][:, 0]

        output['loss_sal'] = self._loss_sal(video_emb, query_emb, video_msk, saliency,
                                            pos_clip)
        return output

    def loss_cal(self, data, output):
        pos_clip = data['pos_clip'][:, 0]

        batch_inds = torch.arange(pos_clip.size(0), device=pos_clip.device)

        coll_v_emb, coll_q_emb = [], []
        for v_emb, q_emb in zip(data['coll_v'], data['coll_q']):
            v_emb_pos = v_emb[batch_inds, pos_clip]
            q_emb_pos = q_emb[:, 0]

            coll_v_emb.append(v_emb_pos)
            coll_q_emb.append(q_emb_pos)

        v_emb = torch.stack(coll_v_emb)
        q_emb = torch.stack(coll_q_emb)
        output['loss_video_cal'] = self._loss_video_cal(v_emb, q_emb)

        v_emb = torch.stack(coll_v_emb, dim=1)
        q_emb = torch.stack(coll_q_emb, dim=1)
        output['loss_layer_cal'] = self._loss_layer_cal(v_emb, q_emb)

        return output

    def loss_neg(self,data,output):
        
        pred = torch.cat([data['relavent_score'],data['neg_relavent_score']],dim=0)
        neg_target = torch.where(data['neg_mask']==1, 0.0, 1.0).unsqueeze(-1)
        pos_target = torch.ones(data['relavent_score'].size(0)).unsqueeze(-1)
        target = torch.cat([pos_target.to(pred.device),neg_target.to(pred.device)],dim=0)
        output['loss_neg'] = self._loss_neg(pred,target)

        return output
    

    def forward(self, data, output):
        if self._loss_reg is not None:
            cls_tgt, reg_tgt = self.get_target(data)
            output = self.loss_reg(data, output, cls_tgt, reg_tgt)
        else:
            cls_tgt = data['saliency']

        if self._loss_cls is not None:
            output = self.loss_cls(data, output, cls_tgt)

        if self._loss_sal is not None:
            output = self.loss_sal(data, output)

        if self._loss_video_cal is not None or self._loss_layer_cal is not None:
            output = self.loss_cal(data, output)

        if self._loss_neg is not None:
            output = self.loss_neg(data,output)

        return output
