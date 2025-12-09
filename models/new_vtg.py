import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import MODELS, build_loss, build_model

from .generator import PointGenerator


def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a['vid']==b['vid']:
            res.append(True)
        else:
            res.append(False)
    return res


@MODELS.register()
class NewVTG(nn.Module):

    def __init__(self,
                 dim_q,
                 dim_v,
                 hidden_dims,
                 strides=(1, 2, 4, 8),
                 buffer_size=1024,
                 max_num_moment=50,
                 merge_cls_sal=True,
                 adapter_cfg=None,
                 pyramid_cfg=None,
                 pooling_cfg=None,
                 class_head_cfg=None,
                 coord_head_cfg=None,
                 loss_cfg=None,
                 use_neg=True):
        super(NewVTG, self).__init__()

        self.adapter = build_model(adapter_cfg,dim_q, dim_v,hidden_dims)
        self.pyramid = build_model(pyramid_cfg, hidden_dims, strides)
        self.pooling = build_model(pooling_cfg, hidden_dims)

        self.class_head = build_model(class_head_cfg, hidden_dims, 1)
        self.coord_head = build_model(coord_head_cfg, hidden_dims, 2)

        self.generator = PointGenerator(strides, buffer_size)

        self.coef = nn.Parameter(torch.ones(len(strides)))
        self.loss = build_loss(loss_cfg)

        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal
        self.use_neg = use_neg

        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(hidden_dims * 2),
            nn.Linear(hidden_dims * 2, 1),)


    def forward(self, data, mode='test'):
        b, t = data['video'].size()[:2]

        video, query = data['video'], data['query']  # [B, Lv, Dv], [B, Lq, Dq]


        video_msk = torch.where(video[:, :, 0].isfinite(), 1, 0)
        video[~video.isfinite()] = 0
        video_ = video.float()

        query_msk = torch.where(query[:, :, 0].isfinite(), 1, 0)
        query[~query.isfinite()] = 0
        query_ = query.float()


        video_emb, query_emb = self.adapter(video_, query_,video_msk, query_msk,mode)
        
        # video_emb: B*T*C
        # query_emb: B*L*C
        # video_msk [B, T]
        # query_msk [B, L]
        relavent_score = self.contrastive_head(
            torch.cat([video_emb.mean(dim=1),query_emb.mean(dim=1)],dim=-1)
        )
        data['relavent_score'] = relavent_score  # [B,1]


        if mode != 'test' and self.use_neg:
            neg_video_ = torch.cat([video_[1:],video[:1]],dim=0)
            neg_video_msk = torch.cat([video_msk[1:],video_msk[:1]],dim=0)

            label = data['label']
            # 生成负样本的label
            neg_label = label[1:] + label[:1]
            neg_mask = torch.tensor(element_wise_list_equal(label, neg_label), 
                                  device=video_.device)  # Create tensor directly on correct device
            neg_mask = neg_mask == False

            neg_video_emb,neg_query_emb = self.adapter(neg_video_,query_,neg_video_msk,query_msk)
            neg_relavent_score = self.contrastive_head(
                torch.cat([neg_video_emb.mean(dim=1),neg_query_emb.mean(dim=1)],dim=-1)
            )
            
            # 使用已存在的 query_emb 做 mean pooling 得到语义表示
            query_feat = query_emb.mean(dim=1)  # [B, C]

            # 视频中的 batch 构造 hard negative：选取 query 相似的其他 video
            sim_matrix = F.cosine_similarity(query_feat.unsqueeze(1), query_feat.unsqueeze(0), dim=-1)  # [B, B]
            sim_matrix.fill_diagonal_(-1e4)  # 排除自己
            topk_inds = sim_matrix.argmax(dim=1)  # 每个 query 选最相似但非自身的 index

            # 构建 hard negatives
            neg_video_ = video_[topk_inds]
            neg_video_msk = video_msk[topk_inds]
            neg_label = [data['label'][i] for i in topk_inds]
            neg_mask = torch.tensor(
                [data['label'][i]['vid'] != neg_label[i]['vid'] for i in range(len(neg_label))],
                device=video_.device
            )

            neg_video_emb, neg_query_emb = self.adapter(neg_video_, query_, neg_video_msk, query_msk)
            hard_neg_score = self.contrastive_head(
                torch.cat([neg_video_emb.mean(dim=1), neg_query_emb.mean(dim=1)], dim=-1)
            )            
            
            hard_neg_mask = torch.tensor(
                [data['label'][i]['vid'] != neg_label[i]['vid'] for i in range(len(neg_label))],
                device=video_.device
            )
            
            data['neg_relavent_score'] = torch.cat([neg_relavent_score, hard_neg_score], dim=0) # [B,1]
            data['neg_mask'] = torch.cat([neg_mask, hard_neg_mask], dim=0) # [B]
            # data['neg_relavent_score'] = neg_relavent_score # [B,1]
            # data['neg_mask'] = neg_mask # [B]
            
            

        pymid, pymid_msk = self.pyramid(video_emb, video_msk, return_mask=mode != 'test')

        # pymid, pymid_msk  # list[B*(T/s)*C], list[B*(T/s)]  s = 1, 2, 4, 8

        point = self.generator(pymid)

        '''
        eg. points = [[0,0,2,1],[1,0,2,1],[2,0,2,1],...,
                      [0,2,4,2],[2,2,4,2],[4,2,4,2],...,
                      [0,4,8,4],[4,4,8,4],[8,4,8,4],...,
                      [0,8,inf,8],[8,8,inf,8],[16,8,inf,8]...]
        '''

        with torch.autocast('cuda', enabled=False):
            video_emb = video_emb.float()
            query_emb = self.pooling(query_emb.float(), query_msk)

            out_class = [self.class_head(e.float()) for e in pymid]
            # out_class = [B*(T/s)*1] s = 1, 2, 4, 8

            out_class = torch.cat(out_class, dim=1)

            if self.coord_head is not None:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                # out_coord = [B*(T/s)*2] s = 1, 2, 4, 8
                out_coord = torch.cat(out_coord, dim=1)
            else:
                out_coord = None

            output = dict(_avg_factor=b)

            if mode != 'test':

                data['point'] = point
                data['video_emb'] = video_emb
                data['query_emb'] = query_emb
                data['video_msk'] = video_msk
                data['pymid_msk'] = pymid_msk
                data['out_class'] = out_class
                data['out_coord'] = out_coord

                output = self.loss(data, output)

            if mode != 'train':
                out_class = out_class.sigmoid()
                out_score = F.cosine_similarity(video_emb, query_emb, dim=-1)

                pyd_shape = [e.size(1) for e in pymid]  # [T, T/2, T/4, T/8]

                if b == 1:
                    output['_out'] = dict(label=data.get('label', [None])[0])

                    pyd_class = out_class[0, :, 0].split(pyd_shape) # [1*(T/s)*1] s = 1, 2, 4, 8

                    saliency = []
                    for shape, score in zip(pyd_shape, pyd_class):
                        if t >= shape:
                            score = score.repeat_interleave(int(t / shape))
                            postfix = score[-1:].repeat(t - score.size(0))
                            score = torch.cat((score, postfix))
                        else:
                            scale = int(shape / t)
                            score = F.max_pool1d(score.unsqueeze(0), scale, stride=scale)[0]
                        saliency.append(score)

                    saliency = torch.stack(saliency).amax(dim=0)  # [T]

                    if self.merge_cls_sal:
                        saliency *= out_score[0]

                    output['_out']['saliency'] = saliency

                    if self.coord_head is not None:
                        boundary = out_coord[0] #  [sum(T/s)*2] s = 1, 2, 4, 8
                        boundary[:, 0] *= -1
                        boundary *= point[:, 3, None].repeat(1, 2)
                        boundary += point[:, 0, None].repeat(1, 2)
                        boundary /= data['fps'][0]
                        boundary = torch.cat((boundary, out_class[0]), dim=-1)

                        _, inds = out_class[0, :, 0].sort(descending=True)
                        boundary = boundary[inds[:self.max_num_moment]]

                        output['_out']['boundary'] = boundary

                        relavent_score_sig = relavent_score.sigmoid()
                        output['_out']['relavent_score'] = relavent_score_sig
                else:
                    outs = []
                    # 使用各自的有效长度而不是全局 t
                    valid_lens = video_msk.sum(dim=1).to(torch.long).tolist()
                    for i in range(b):
                        lbl = data.get('label', [None]*b)[i]
                        fps_i = data['fps'][i]
                        ti = valid_lens[i]

                        pyd_class_i = out_class[i, :, 0].split(pyd_shape)
                        saliency_i = []
                        for shape, score in zip(pyd_shape, pyd_class_i):
                            if ti >= shape:
                                rep = max(1, int(ti / shape))
                                score = score.repeat_interleave(rep)
                                if score.size(0) < ti:
                                    postfix = score[-1:].repeat(ti - score.size(0))
                                    score = torch.cat((score, postfix))
                                else:
                                    score = score[:ti]
                            else:
                                scale = max(1, int(shape / max(1, ti)))
                                score = F.max_pool1d(score.unsqueeze(0), scale, stride=scale)[0]
                                score = score[:ti]
                            saliency_i.append(score)
                        saliency_i = torch.stack(saliency_i).amax(dim=0)  # [Ti]

                        if self.merge_cls_sal:
                            # out_score[i] has length T (padded); match per-sample valid length ti
                            out_score_i = out_score[i][:ti]
                            saliency_i = saliency_i * out_score_i

                        pred_i = dict(label=lbl, saliency=saliency_i)

                        if self.coord_head is not None:
                            boundary_i = out_coord[i]
                            boundary_i[:, 0] *= -1
                            boundary_i = boundary_i * point[:, 3, None].repeat(1, 2)
                            boundary_i = boundary_i + point[:, 0, None].repeat(1, 2)
                            boundary_i = boundary_i / fps_i
                            boundary_i = torch.cat((boundary_i, out_class[i]), dim=-1)

                            _, inds = out_class[i, :, 0].sort(descending=True)
                            boundary_i = boundary_i[inds[:self.max_num_moment]]
                            pred_i['boundary'] = boundary_i

                            pred_i['relavent_score'] = relavent_score[i].sigmoid()

                        outs.append(pred_i)

                    # 返回批量样本的结果列表，交由上游 EvalHook 展开
                    output['_out'] = outs
                

        return output
