import os
import numpy as np
import joblib
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
import argparse


def set_seed(seed=42):
    """固定随机种子以保证可复现性"""
    np.random.seed(seed)


class KMeansPicker:
    """统一的 KMeans 特征提取器，支持 .npz 和 .pt 文件"""
    
    def __init__(self, folder, n_clusters, file_type='npz', array_key=None, kmeans_file=None):
        """
        Args:
            folder: 存放特征文件的文件夹路径
            n_clusters: 聚类的簇数
            file_type: 文件类型，'npz' 或 'pt'
            array_key: npz 文件中的数组键名（仅 npz 类型需要）
            kmeans_file: 可选，保存/加载 kmeans 模型的文件夹
        """
        self.folder = folder
        self.n_clusters = n_clusters
        self.file_type = file_type.lower()
        self.array_key = array_key
        self.feat_dicts = {
            'max_pool_feats': [],
            'mean_pool_feats': []
        }
        
        # 加载特征文件
        self._load_files()
        
        # 加载或训练 KMeans 模型
        if kmeans_file and os.path.exists(os.path.join(kmeans_file, 'max_pool_feats.pkl')) \
                and os.path.exists(os.path.join(kmeans_file, 'mean_pool_feats.pkl')):
            print(f"Loading KMeans models from {kmeans_file}...")
            self.kmeans_model_dict = {
                'max_pool_feats': joblib.load(os.path.join(kmeans_file, 'max_pool_feats.pkl')),
                'mean_pool_feats': joblib.load(os.path.join(kmeans_file, 'mean_pool_feats.pkl')),
            }
        else:
            self.kmeans_model_dict = self._train_kmeans_models()
            if kmeans_file:
                self.save_kmeans_models(kmeans_file)

    def _load_files(self):
        """根据文件类型加载特征文件"""
        if self.file_type == 'npz':
            self._load_npz_files()
        elif self.file_type == 'pt':
            self._load_pt_files()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}. Use 'npz' or 'pt'.")

    def _load_npz_files(self):
        """加载 .npz 文件"""
        if self.array_key is None:
            raise ValueError("array_key is required for npz files")
        
        for file_name in os.listdir(self.folder):
            if file_name.endswith('.npz'):
                file_path = os.path.join(self.folder, file_name)
                data = np.load(file_path)
                array = data[self.array_key]
                self.feat_dicts['max_pool_feats'].append(np.max(array, axis=0))
                self.feat_dicts['mean_pool_feats'].append(np.mean(array, axis=0))
        
        print(f"Loaded {len(self.feat_dicts['max_pool_feats'])} npz files")

    def _load_pt_files(self):
        """加载 .pt 文件"""
        for file_name in os.listdir(self.folder):
            if file_name.endswith('.pt'):
                file_path = os.path.join(self.folder, file_name)
                tensor = torch.load(file_path, map_location='cpu')
                
                # 忽略 dict、list、其他复杂结构
                if not isinstance(tensor, torch.Tensor):
                    print(f"Skipping {file_name}: Not a tensor")
                    continue
                
                array = tensor.numpy()
                self.feat_dicts['max_pool_feats'].append(np.max(array, axis=0))
                self.feat_dicts['mean_pool_feats'].append(np.mean(array, axis=0))
        
        print(f"Loaded {len(self.feat_dicts['max_pool_feats'])} pt files")

    def _train_kmeans_models(self):
        """训练 KMeans 模型"""
        kmeans_model_dict = {}
        for k, v in self.feat_dicts.items():
            print(f"Training KMeans for {k}...")
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(v)
            kmeans_model_dict[k] = kmeans
        return kmeans_model_dict

    def random_pick_features(self):
        """从每个簇中随机选取一个特征"""
        random_feat_dicts = defaultdict(list)
        for k in self.feat_dicts:
            kmeans = self.kmeans_model_dict[k]
            for cluster_label in np.unique(kmeans.labels_):
                cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
                random_index = np.random.choice(cluster_indices)
                sample = self.feat_dicts[k][random_index]
                random_feat_dicts[k].append(sample)
        return random_feat_dicts

    def save_features(self, target_file):
        """保存聚类特征到文件"""
        random_feat_dicts = self.random_pick_features()
        np.savez(target_file,
                 max_pool_c=np.array(random_feat_dicts['max_pool_feats']),
                 mean_pool_c=np.array(random_feat_dicts['mean_pool_feats']))
        print(f"Saved features to {target_file}")

    def save_kmeans_models(self, kmeans_file):
        """保存 KMeans 模型"""
        if not os.path.exists(kmeans_file):
            os.makedirs(kmeans_file)
        joblib.dump(self.kmeans_model_dict['max_pool_feats'], 
                    os.path.join(kmeans_file, 'max_pool_feats.pkl'))
        joblib.dump(self.kmeans_model_dict['mean_pool_feats'], 
                    os.path.join(kmeans_file, 'mean_pool_feats.pkl'))
        print(f"Saved KMeans models to {kmeans_file}")


def main():
    parser = argparse.ArgumentParser(description='KMeans 聚类并提取特征（支持 npz 和 pt 文件）')
    parser.add_argument('--folder', type=str, required=True, 
                        help='存放特征文件的文件夹路径')
    parser.add_argument('--target_file', type=str, required=True, 
                        help='保存聚类特征的目标文件')
    parser.add_argument('--n_clusters', type=int, required=True, 
                        help='聚类的簇数')
    parser.add_argument('--file_type', type=str, default='npz', choices=['npz', 'pt'],
                        help='文件类型: npz 或 pt (默认: npz)')
    parser.add_argument('--array_key', type=str, default=None,
                        help='npz 文件中的数组键名（仅 npz 类型需要）')
    parser.add_argument('--kmeans_file', type=str, default=None, 
                        help='可选，保存/加载 kmeans 模型的文件夹')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于保证可复现性 (默认: 42)')
    args = parser.parse_args()

    # 固定随机种子
    set_seed(args.seed)

    # 验证参数
    if args.file_type == 'npz' and args.array_key is None:
        parser.error("--array_key is required when --file_type is 'npz'")

    picker = KMeansPicker(
        folder=args.folder,
        n_clusters=args.n_clusters,
        file_type=args.file_type,
        array_key=args.array_key,
        kmeans_file=args.kmeans_file
    )
    picker.save_features(args.target_file)


if __name__ == '__main__':
    main()
