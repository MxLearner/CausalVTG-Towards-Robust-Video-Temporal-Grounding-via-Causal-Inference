import math
import random
import nncore
import numpy as np
import torch
import torchvision.transforms as T
from nncore.dataset import DATASETS, Dataset
from nncore.ops import temporal_iou
from nncore.parallel import DataContainer

from .eval import vtg_eval


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

@DATASETS.register()
class NewGrounding(Dataset):

    def __init__(self,
                 label_path,
                 video_path=None,
                 cache_path=None,
                 query_path=None,
                 min_video_len=-1,
                 max_video_len=-1,
                 max_saliency=12,
                 fps=None,
                 unit=None,
                 normalize_q = True,
                 normalize_v = True,
                 preload=False,
                 preload_device='cpu',
                 preload_dtype=None):
        """Dataset for VTG with optional in-memory preloading.

        Args:
            label_path (str): Path to label json/jsonl.
            cache_path (List[str]): List of directories containing per-video feature npz/pt files.
            query_path (str): Directory containing per-query feature files.
            preload (bool): If True, load all video and query features into RAM at init.
            preload_device (str): 'cpu', 'cuda', or 'cuda:N'. Only used if preload=True.
            preload_dtype (torch.dtype|None): Optional dtype cast for preloaded tensors (e.g., torch.float16).
        """
        assert fps is not None

        label = nncore.load(label_path)
        self.label = []
        for anno in label:
            if min_video_len > 0 and float(anno['duration']) < min_video_len:
                continue
            if max_video_len > 0 and float(anno['duration']) > max_video_len:
                continue
            if 'tacos' in label_path:
                anno['vid'] = f"{anno['vid']}-cam-002"
            self.label.append(anno)

        self.label_path = label_path
        self.video_path = video_path
        self.cache_path = cache_path
        self.query_path = query_path
        self.min_video_len = min_video_len
        self.max_video_len = max_video_len
        self.max_saliency = max_saliency
        self.fps = fps
        self.unit = unit
        self.normalize_q = normalize_q
        self.normalize_v = normalize_v
        self.preload = preload
        # Resolve device and dtype for preloading
        try:
            _req_device = torch.device(preload_device)
        except Exception:
            _req_device = torch.device('cpu')
        if _req_device.type == 'cuda' and not torch.cuda.is_available():
            _req_device = torch.device('cpu')
        self._preload_device = _req_device
        self._preload_dtype = preload_dtype

        # Internal caches when preloading
        self._video_cache = {}
        self._query_cache = {}

        if self.preload:
            self._preload_data()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]

        data = dict()
        if self.preload:
            # Fetch from cache (assumed already normalized and concatenated)
            vid = label['vid']
            qid = label['qid']
            video_tensor = self._video_cache[vid]
            query_tensor = self._query_cache[qid]
            data['video'] = DataContainer(video_tensor, pad_value=float('inf'))
            data['query'] = DataContainer(query_tensor, pad_value=float('inf'))
        else:
            data = self.bind_video(label, data)
            data = self.bind_query(label, data)

        if 'relevant_windows' in label:
            data = self.bind_boundary(label, data)

        if 'relevant_windows' in label or 'saliency_scores' in label:
            data = self.bind_saliency(label, data)

        if 'no_answer' in label :
            data = self.bind_no_answer(label,data)

        data['label'] = DataContainer(label, cpu_only=True)
        data['fps'] = DataContainer(self.fps, cpu_only=True)

        return data
    
    def bind_no_answer(self, label, data):
        # 将 no_answer 转换为 Tensor
        no_answer = torch.tensor(label['no_answer'], dtype=torch.bool)
        # 使用 torch.where 进行条件判断
        no_answer = torch.where(no_answer, 1, 0)
        data['no_answer'] = no_answer
        return data
        

    def bind_video(self, label, data):
        vid = label['vid']
        video = None
        video_list = []
        for path in self.cache_path:
            if '/clip' in path:
                video = (np.load(nncore.join(path, f'{vid}.npz'))['features']
                         .astype(np.float32))
            elif 'slowfast' in path:
                video = (np.load(nncore.join(path, f'{vid}.npz'))['features']
                         .astype(np.float32))
            elif 'internvideo2' in path:
                video = torch.load(nncore.join(path, f'{vid}.pt')).float().numpy()
            else:
                assert ValueError(f"Unknown cache path: {path}")
            
            assert video is not None
            if self.normalize_v:
                video = l2_normalize_np_array(video)
            video_list.append(video)

        min_len = min([len(e) for e in video_list])
        v_feat_list = [e[:min_len] for e in video_list]
        video = np.concatenate(v_feat_list, axis=-1)
        video = torch.from_numpy(video).view(video.shape[0], -1)
        data['video'] = DataContainer(video, pad_value=float('inf')) # [Lv, Dv]
        return data

    def bind_query(self, label, data):
        qid = label['qid']
        query = None

        if 'internvideo2'in self.query_path:
            query = torch.load(nncore.join(self.query_path, f'qid{qid}.pt')).float().numpy()

        elif 'clip_text' in self.query_path:
            query = (np.load(nncore.join(self.query_path, f'qid{qid}.npz'))["last_hidden_state"]
                     .astype(np.float32))

        else:
            raise ValueError(f"Unknown query path: {self.query_path}")
        assert query is not None

        if self.normalize_q:
            query = l2_normalize_np_array(query)
        query = torch.from_numpy(query).view(query.shape[0], -1)
        data['query'] = DataContainer(query, pad_value=float('inf')) # [Lq, Dq]
        return data

    def _preload_data(self):
        """Preload all video and query features into memory.
        This can significantly reduce IO overhead during training/evaluation
        at the cost of additional RAM usage.
        """
        # Preload videos
        for anno in self.label:
            vid = anno['vid']
            if vid in self._video_cache:
                continue
            video = None
            video_list = []
            for path in self.cache_path:
                if '/clip' in path:
                    video = (np.load(nncore.join(path, f'{vid}.npz'))['features']
                             .astype(np.float32))
                elif 'slowfast' in path:
                    video = (np.load(nncore.join(path, f'{vid}.npz'))['features']
                             .astype(np.float32))
                elif 'internvideo2' in path:
                    video = torch.load(nncore.join(path, f'{vid}.pt')).float().numpy()
                else:
                    raise ValueError(f"Unknown cache path: {path}")
                assert video is not None
                if self.normalize_v:
                    video = l2_normalize_np_array(video)
                video_list.append(video)
            min_len = min([len(e) for e in video_list])
            v_feat_list = [e[:min_len] for e in video_list]
            video = np.concatenate(v_feat_list, axis=-1)
            video_tensor = torch.from_numpy(video).view(video.shape[0], -1).contiguous()
            if self._preload_dtype is not None:
                video_tensor = video_tensor.to(dtype=self._preload_dtype)
            # Move to target device if requested
            video_tensor = video_tensor.to(self._preload_device, non_blocking=True)
            self._video_cache[vid] = video_tensor

        # Preload queries
        for anno in self.label:
            qid = anno['qid']
            if qid in self._query_cache:
                continue
            if 'internvideo2' in self.query_path:
                query = torch.load(nncore.join(self.query_path, f'qid{qid}.pt')).float().numpy()
            elif 'clip_text_features' in self.query_path:
                query = (np.load(nncore.join(self.query_path, f'qid{qid}.npz'))["last_hidden_state"]
                         .astype(np.float32))
            else:
                raise ValueError(f"Unknown query path: {self.query_path}")
            if self.normalize_q:
                query = l2_normalize_np_array(query)
            query_tensor = torch.from_numpy(query).view(query.shape[0], -1).contiguous()
            if self._preload_dtype is not None:
                query_tensor = query_tensor.to(dtype=self._preload_dtype)
            query_tensor = query_tensor.to(self._preload_device, non_blocking=True)
            self._query_cache[qid] = query_tensor

        # Optional: print a brief summary (could be replaced by logger)
        dtype_bytes = 2 if self._preload_dtype in (torch.float16, torch.bfloat16) else 4
        total_video_mb = sum(v.numel() * dtype_bytes for v in self._video_cache.values()) / (1024 ** 2)
        total_query_mb = sum(q.numel() * dtype_bytes for q in self._query_cache.values()) / (1024 ** 2)
        device_str = str(self._preload_device)
        print(
            f"[NewGrounding] Preloaded {len(self._video_cache)} videos (~{total_video_mb:.2f} MB) "
            f"and {len(self._query_cache)} queries (~{total_query_mb:.2f} MB) on {device_str}"
            f" with dtype={self._preload_dtype or 'float32'}."
        )

    def bind_boundary(self, label, data):
        max_time = data['video'].data.size(0) / self.fps

        boundary = torch.Tensor(label['relevant_windows'])
        boundary = boundary.clamp(min=0, max=max_time)

        inds = boundary[:, 1] - boundary[:, 0] < 0
        for idx in inds.nonzero()[:, 0]:
            boundary[idx] = boundary[idx].roll(1)

        inds = boundary[:, 1] - boundary[:, 0] < 1 / self.fps
        for idx in inds.nonzero()[:, 0]:
            center = boundary[idx].sum() / 2
            center = center.clamp(min=0.5 / self.fps, max=max_time - 0.5 / self.fps)
            boundary[idx, 0] = center - 0.5 / self.fps
            boundary[idx, 1] = center + 0.5 / self.fps

        data['boundary'] = DataContainer(boundary, pad_value=float('inf'))
        return data

    def bind_saliency(self, label, data):
        num_clips = data['video'].data.size(0)

        if 'saliency_scores' in label and 'relevant_clip_ids' in label:
            saliency = torch.zeros(int(label['duration'] * self.fps))
            for idx, score in zip(label['relevant_clip_ids'], label['saliency_scores']):
                saliency[idx] = sum(score) / self.max_saliency
            assert saliency.size(0) >= num_clips and saliency.size(0) - num_clips < 5 
            saliency = saliency[:num_clips]
        else:
            boundary_ind = data['boundary'].data * self.fps
            pos_clips = []
            for bnd in boundary_ind.tolist():
                pos_clips += list(range(math.ceil(bnd[0]), math.ceil(bnd[1])))
            assert len(pos_clips) > 0
            saliency = torch.zeros(num_clips)
            saliency[pos_clips] = 1

        pos_clip = random.sample(saliency.nonzero()[:, 0].tolist(), 1) # saliency.nonzero()[:, 0]  取非0元素索引
        pos_clip = torch.LongTensor(pos_clip)

        data['saliency'] = DataContainer(saliency)
        data['pos_clip'] = DataContainer(pos_clip)

        return data
    def evaluate(self,
                 blobs,
                 nms_cfg=dict(type='normal', thres=0.7),
                 dump_template=None,
                 logger=None):

        # Flatten possible nested outputs when model returns list per batch
        def _flatten_blobs(items):
            flat = []
            for it in items:
                if isinstance(it, (list, tuple)):
                    for e in it:
                        if isinstance(e, dict) and '_out' in e:
                            flat.append(e['_out'])
                        else:
                            flat.append(e)
                elif isinstance(it, dict) and '_out' in it:
                    flat.append(it['_out'])
                else:
                    flat.append(it)
            return flat

        blobs = _flatten_blobs(blobs)
        
        out = []
        for blob in blobs:
            pred = dict(vid=blob['label']['vid'], qid=blob['label']['qid'])

            if 'boundary' in blob:
                bnd = blob['boundary']

                # 1. clamp regression ranges
                inds = bnd[:, 1] < blob['label']['duration']
                bnd[:, :2] = bnd[:, :2].clamp(min=0, max=blob['label']['duration'])

                # 2. round boundaries to units
                if self.unit is not None and self.unit > 0:
                    bnd[inds, :2] = torch.round(bnd[inds, :2] / self.unit) * self.unit   # 调整为倍数

                # 3. perform nms (vectorized/optimized with optional pre_topk)
                if nms_cfg is not None:
                    assert nms_cfg['type'] in ('normal', 'linear', 'gaussian')

                    scores = bnd[:, -1]
                    pre_topk = nms_cfg.get('pre_topk', None)
                    if isinstance(pre_topk, int) and pre_topk > 0 and bnd.size(0) > pre_topk:
                        top_scores, top_idx = scores.topk(pre_topk, largest=True, sorted=True)
                        bnd = bnd[top_idx]
                        scores = top_scores

                    ntype = nms_cfg['type']
                    if ntype == 'normal':
                        # Greedy NMS with fixed initial sort (equivalent since scores不更新)
                        order = scores.argsort(descending=True)
                        bnd_sorted = bnd[order]
                        keep_indices = []
                        if bnd_sorted.size(0) > 0:
                            cur = 0
                            while cur < bnd_sorted.size(0):
                                keep_indices.append(cur)
                                if cur + 1 >= bnd_sorted.size(0):
                                    break
                                iou = temporal_iou(bnd_sorted[cur:cur+1, :-1], bnd_sorted[cur+1:, :-1])[0]
                                th = nms_cfg.get('thres', 0.7)
                                mask = iou < th
                                # 保留当前，过滤掉高 IoU 的后续候选
                                if mask.any():
                                    # 拼接保留的元素
                                    kept_rest = bnd_sorted[cur+1:][mask]
                                    bnd_sorted = torch.cat([bnd_sorted[:cur+1], kept_rest], dim=0)
                                else:
                                    bnd_sorted = bnd_sorted[:cur+1]
                                cur += 1
                        bnd = bnd_sorted
                        # 最终再按分数排序（稳定）
                        if bnd.size(0) > 0:
                            _, order2 = bnd[:, -1].sort(descending=True)
                            bnd = bnd[order2]
                    else:
                        # Linear/Gaussian: 分数在过程中会衰减，需要每轮选择当前最大分数
                        K = bnd.size(0)
                        if K > 0:
                            bnd_scores = bnd[:, -1].clone()
                            processed = torch.zeros(K, dtype=torch.bool, device=bnd.device)
                            order_list = []
                            th = nms_cfg.get('thres', 0.7)
                            sigma = nms_cfg.get('sigma', 0.5)
                            for _ in range(K):
                                # 选当前未处理里分数最高的一个
                                cur_mask = ~processed
                                if not cur_mask.any():
                                    break
                                cur_rel = torch.where(cur_mask, bnd_scores, torch.tensor(float('-inf'), device=bnd.device, dtype=bnd_scores.dtype))
                                cur_idx = int(cur_rel.argmax().item())
                                order_list.append(cur_idx)
                                processed[cur_idx] = True

                                # 对其余未处理元素做分数衰减
                                rest_mask = ~processed
                                rest_mask[cur_idx] = False
                                if rest_mask.any():
                                    rest_idx = rest_mask.nonzero(as_tuple=False).squeeze(1)
                                    iou = temporal_iou(bnd[cur_idx:cur_idx+1, :-1], bnd[rest_idx, :-1])[0]
                                    if ntype == 'linear':
                                        bnd_scores[rest_idx] = bnd_scores[rest_idx] * (1 - iou)
                                    else:  # gaussian
                                        bnd_scores[rest_idx] = bnd_scores[rest_idx] * torch.exp(-(iou ** 2) / sigma)

                            # 最终按更新后的分数排序
                            new_scores, final_order = bnd_scores.sort(descending=True)
                            bnd[:, -1] = new_scores
                            bnd = bnd[final_order]

                pred['pred_relevant_windows'] = bnd.tolist()

                if 'relavent_score' in blob:
                    pred['relavent_score'] = blob['relavent_score']

            if 'saliency' in blob:
                pred['pred_saliency_scores'] = blob['saliency'].tolist()

            out.append(pred)

        if dump_template is not None:
            # 确保所有数据可JSON序列化
            json_out = []
            for pred in out:
                json_pred = {
                    'vid': pred['vid'],
                    'qid': pred['qid'],
                    'pred_relevant_windows': pred.get('pred_relevant_windows', []),
                    'pred_saliency_scores': pred.get('pred_saliency_scores', [])
                }
                if 'relavent_score' in pred:
                    json_pred['relavent_score'] = float(pred['relavent_score'].item() if torch.is_tensor(pred['relavent_score']) else pred['relavent_score'])
                json_out.append(json_pred)
            
            out_path = dump_template.format('val' if 'val' in self.label_path else 'test')
            logger.info(f'Dumping inference outputs to {out_path}...')
            nncore.dump(json_out, out_path)
            return

        label = nncore.load(self.label_path)
        results = vtg_eval(out, label)['brief']

        return results
