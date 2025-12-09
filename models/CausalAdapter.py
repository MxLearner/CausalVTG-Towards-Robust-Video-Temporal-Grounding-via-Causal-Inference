import math
import os
import numpy as np
import torch
import torch.nn as nn
from nncore.nn import MODELS, build_model

import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    # print(mask.shape)
    return inputs + (1.0 - mask) * mask_value

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.size()
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # b * 8 * 100 * 32
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # b * 8 * 100 * 32
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # b * 8 * 100 * 32

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) #  b * 8 * 1 * 32  mul  b * 8 * 32 * 1   -> b * 8 * 1
        
        if attention_mask is not None:
            # 将[B, T]的mask转换为[B, 1, T, 1]并转换为很大的负数
            attention_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(-1)) * -1e9
            scores = scores + attention_mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        out = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return out

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        '''
        hidden_states  torch.Size([64, 100, 256])  b * len * dim
        encoder_hidden_states  torch.Size([1, 512, 256]) 1 * num_cluster * dim 
        attention_mask  b * len
        ''' 
        batch_size, seq_len, hidden_size = hidden_states.size()
        Q = self.query(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(encoder_hidden_states).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(encoder_hidden_states).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # b * 8 * 100 * 512
        
        if attention_mask is not None:
            # 将[B, T]的mask转换为[B, 1, T, 1]并转换为很大的负数
            attention_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(-1)) * -1e9
            scores = scores + attention_mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        out = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return out

class FeatureFusion(nn.Module):
    def __init__(self, hidden_size):
        super(FeatureFusion, self).__init__()
        self.aug_linear = nn.Linear(hidden_size, 1)
        self.ori_linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_feats, fused_feats):
        aug_linear_weight = self.aug_linear(fused_feats)
        ori_linear_weight = self.ori_linear(local_feats)
        aug_weight = self.sigmoid(aug_linear_weight + ori_linear_weight)
        out_feats = torch.mul(aug_weight, fused_feats) + torch.mul((1 - aug_weight), local_feats)
        return out_feats

class FrontDoorEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads=1, dropout=0.1):
        super(FrontDoorEncoder, self).__init__()

        assert(hidden_size % num_heads == 0), "hidden_size must be divisible by num_heads"
        self.self_attn1 = SelfAttention(hidden_size, num_heads, dropout)
        # self.self_attn2 = SelfAttention(hidden_size, num_heads, dropout)
        # self.self_attn3 = SelfAttention(hidden_size, num_heads, dropout)
        # self.cross_attn1 = CrossAttention(hidden_size, num_heads, dropout)
        self.cross_attn2 = CrossAttention(hidden_size, num_heads, dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fusion = FeatureFusion(hidden_size)

        # # 新增线性变换层
        # self.concat_proj = nn.Sequential(
        #     nn.Linear(2 * hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_size)
        # )

    def forward(self, local_feats,  mean_pool_feats, local_masks=None):
        '''
        local_feats : batch_size, seq_len, hidden_size
        mean_pool_feats : batch_size, num_clusters, hidden_size
        '''
        # Self-attention on local features
        local_self_attn = self.self_attn1(local_feats, attention_mask=local_masks)
        # mean_self_attn = self.self_attn3(mean_pool_feats)


        # Cross-attention between local features and mean pooling features
        local_mean_cross_attn = self.cross_attn2(hidden_states=local_feats, 
        encoder_hidden_states=mean_pool_feats,attention_mask=local_masks)

        # Combine the cross-attention results
        combined_feats =  local_mean_cross_attn

        # Layer normalization
        # combined_feats = self.ln(combined_feats)

        # Feature fusion
        out_feats = self.fusion(local_feats, combined_feats)

        out_feats = self.ln(out_feats)
        # # 直接concat后线性变换
        # concat_feats = torch.cat([local_feats, combined_feats], dim=-1)
        # out_feats = self.concat_proj(concat_feats)

        return out_feats

@MODELS.register()
class CausalAdapter(nn.Module):

    def __init__(self,
                 dim_q,
                 dim_v,
                 hidden_dims,
                 dropout=0.5,
                 use_tef=True,
                 pos_cfg=None,
                 tem_cfg=None,
                 video_cluster_path_list=None,
                 query_cluster_path=None,
                 num_clusters=256):
        super(CausalAdapter,self).__init__()

        self.video_map = nn.Sequential(
            nn.LayerNorm((dim_v + 2) if use_tef else dim_v),
            nn.Dropout(dropout),
            nn.Linear((dim_v + 2) if use_tef else dim_v, hidden_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims))

        self.query_map = nn.Sequential(
            nn.LayerNorm(dim_q),
            nn.Dropout(dropout),
            nn.Linear(dim_q, hidden_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims))

        self.pos = build_model(pos_cfg, dims=hidden_dims)
        self.tem = build_model(tem_cfg, dims=hidden_dims)

        self.dim_v = dim_v
        self.dim_q = dim_q
        self.hidden_dims = hidden_dims

        self.dropout = dropout
        self.use_tef = use_tef

        # 前门：记忆系统作为非训练参数（缓冲区）
        # 聚类路径仅用于初始化原始特征，经过 map_cluster_mean 后才存储为记忆系统

        # 处理视频聚类中心初始化
        if video_cluster_path_list is not None and len(video_cluster_path_list) > 0:
            loaded_video_clusters = []
            for video_cluster_path in video_cluster_path_list:
                assert os.path.exists(video_cluster_path), f"视频聚类路径 {video_cluster_path} 不存在"
                npz_obj = np.load(video_cluster_path)
                if 'mean_pool_c' in npz_obj:
                    arr = npz_obj['mean_pool_c']  # [num_clusters, dim_v]
                else:  # 兼容直接存 ndarray 的情况
                    arr = npz_obj
                loaded_video_clusters.append(torch.from_numpy(arr).float().unsqueeze(0))  # [1, num_clusters, dim_v]
            # 多个视频聚类使用特征维度拼接: [1, num_clusters, dim_v * N]
            video_cluster_init = torch.cat(loaded_video_clusters, dim=-1)  # [1, num_clusters, dim_v * N]
        else:
            # 随机初始化
            video_cluster_init = torch.randn(1, num_clusters, self.dim_v)
            nn.init.xavier_uniform_(video_cluster_init)
        video_cluster_in_dim = video_cluster_init.shape[-1]

        # 处理文本/查询聚类中心初始化
        if query_cluster_path is not None and os.path.exists(query_cluster_path):
            query_npz = np.load(query_cluster_path)
            if 'mean_pool_c' in query_npz:
                q_arr = query_npz['mean_pool_c']  # [num_clusters, dim_q]
            else:
                q_arr = query_npz
            query_cluster_init = torch.from_numpy(q_arr).float().unsqueeze(0)  # [1, num_clusters, dim_q]
        else:
            query_cluster_init = torch.randn(1, num_clusters, self.dim_q)
            nn.init.xavier_uniform_(query_cluster_init)
        query_cluster_in_dim = query_cluster_init.shape[-1]

        # 定义 map_cluster_mean 网络层（可训练）
        self.video_map_cluster_mean = nn.Sequential(
            nn.LayerNorm(video_cluster_in_dim),
            nn.Dropout(dropout),
            nn.Linear(video_cluster_in_dim, hidden_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims))        

        self.query_map_cluster_mean = nn.Sequential(
            nn.LayerNorm(query_cluster_in_dim),
            nn.Dropout(dropout),
            nn.Linear(query_cluster_in_dim, hidden_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims))

        # 初始化记忆系统：将初始聚类中心经过 map_cluster_mean 后存储
        with torch.no_grad():
            video_memory_init = self.video_map_cluster_mean(video_cluster_init)  # [1, num_clusters, hidden_dims]
            query_memory_init = self.query_map_cluster_mean(query_cluster_init)  # [1, num_clusters, hidden_dims]
        
        # 注册为 buffer（非训练参数，但会保存/加载）
        self.register_buffer('video_memory', video_memory_init)
        self.register_buffer('query_memory', query_memory_init)

        self.video_front_door_encoder = FrontDoorEncoder(hidden_size=hidden_dims, num_heads=8, dropout=0.1)
        self.query_front_door_encoder = FrontDoorEncoder(hidden_size=hidden_dims, num_heads=8, dropout=0.1)

        self.cq_attention = CQAttention(dim=hidden_dims, drop_rate=dropout)


    def forward(self, video_emb, query_emb, video_msk, query_msk, mode=None):
        # video_emb: B * Lv * Dv
        # query_emb: B * Lq * Dq
        # video_msk: B * Lv
        # query_msk: B * Lq

        b, t, _ = video_emb.size()

        video_emb1 = video_emb.clone()

        if self.use_tef:
            tef_s = torch.arange(0, t,  device=video_emb.device)/t
            tef_e = tef_s + 1.0 / t
            tef = torch.stack((tef_s, tef_e), dim=1)
            tef = tef.repeat(b, 1, 1)  # B * Lv * 2
            video_emb = torch.cat((video_emb, tef), dim=-1)


        v_emb = self.video_map(video_emb)  # B * Lv * hidden_dims
        q_emb = self.query_map(query_emb)  # B * Lq * hidden_dims

        # 1. 对当前 batch 样本进行全局平均池化
        pooled_v = video_emb1.mean(dim=1)  # [B, dim_v]
        pooled_q = query_emb.mean(dim=1)   # [B, dim_q]
        
        # 2. 经过 map_cluster_mean 网络
        pooled_v_mapped = self.video_map_cluster_mean(pooled_v)  # [B, hidden_dims]
        pooled_q_mapped = self.query_map_cluster_mean(pooled_q)  # [B, hidden_dims]
        
        # 3. 在训练模式下，通过余弦相似度更新记忆系统
        if mode == 'train':
            with torch.no_grad():
                # 视频记忆更新
                v_memory = self.video_memory.squeeze(0)  # [num_clusters, hidden_dims]
                # 计算余弦相似度
                pooled_v_norm = F.normalize(pooled_v_mapped, p=2, dim=1)  # [B, hidden_dims]
                v_memory_norm = F.normalize(v_memory, p=2, dim=1)  # [num_clusters, hidden_dims]
                v_sim = torch.matmul(pooled_v_norm, v_memory_norm.T)  # [B, num_clusters]
                
                # 找到最相似的记忆索引
                v_max_sim_idx = torch.argmax(v_sim, dim=1)  # [B]
                
                # 更新记忆：(old + new) / 2
                for i, idx in enumerate(v_max_sim_idx):
                    v_memory[idx] = (v_memory[idx] + pooled_v_mapped[i]) / 2.0
                
                self.video_memory = v_memory.unsqueeze(0)
                
                # 查询记忆更新
                q_memory = self.query_memory.squeeze(0)  # [num_clusters, hidden_dims]
                pooled_q_norm = F.normalize(pooled_q_mapped, p=2, dim=1)  # [B, hidden_dims]
                q_memory_norm = F.normalize(q_memory, p=2, dim=1)  # [num_clusters, hidden_dims]
                q_sim = torch.matmul(pooled_q_norm, q_memory_norm.T)  # [B, num_clusters]
                
                q_max_sim_idx = torch.argmax(q_sim, dim=1)  # [B]
                
                for i, idx in enumerate(q_max_sim_idx):
                    q_memory[idx] = (q_memory[idx] + pooled_q_mapped[i]) / 2.0
                
                self.query_memory = q_memory.unsqueeze(0)
        
        # 4. 使用当前记忆系统进入 FrontDoorEncoder
        v_emb = self.video_front_door_encoder(v_emb, self.video_memory, video_msk)
        q_emb = self.query_front_door_encoder(q_emb, self.query_memory, query_msk)
            
        fused_features = self.cq_attention(
            v_emb, 
            q_emb,
            c_mask=video_msk,
            q_mask=query_msk
        )
        v_emb = fused_features # + v_emb   


        v_pe = self.pos(v_emb)
        v_emb = self.tem(v_emb, q_emb, q_pe=v_pe, q_mask=video_msk, k_mask=query_msk)

        return v_emb, q_emb
