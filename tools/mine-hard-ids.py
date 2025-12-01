import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from freeze_torch_model import load_torchWRN_model, MarsSmall128

class ReIDListDataset(Dataset):
    def __init__(self, root_dir, list_path, transform=None, relabel=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if 3 == len(parts):
                    img_path, pid, cam_id = parts
                else:
                    raise ValueError(f"Invalid line format in {list_path}: {line}")

                self.samples.append((img_path, int(pid), int(cam_id)))

        # Map_ids to class indices starting from 0
        if relabel:
            label_map = {pid: idx for idx, pid in enumerate(sorted(set(pid for _, pid, _ in self.samples)))}
            self.relabel(label_map)

    def relabel(self, label_map):
        self.label_map = label_map
        new_samples = []
        for img_path, pid, cam_id in self.samples:
            if pid in self.label_map:
                new_label = self.label_map[pid]
                new_samples.append((img_path, new_label, cam_id))
            else:
                print(f"PID {pid} not in gallery, skipping")
                continue

        self.samples = new_samples
        self.classes = list(self.label_map.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, camid = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)

        return img, label, camid

class ToNumpyUint8:
    def __call__(self, img):
        arr = np.array(img)
        arr = arr[:, :, ::-1].copy() # from RGB to BGR
        return arr

def load_dataset(cfg, image_shape):
    path = cfg.dataset
    batch_size = cfg.batch_sz
    map_file = cfg.map

    model_file =  cfg.model

    if model_file.endswith(".pb"):
        trans_qg = transforms.Compose([
            transforms.Resize(image_shape),
            ToNumpyUint8(),
        ])
    elif model_file.endswith(".pth"):
        trans_qg = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:x*255),
        ])


    dataset = ReIDListDataset(path, f"{path}/{map_file}", transform=trans_qg, relabel=False)
    loader = DataLoader(dataset, batch_size=batch_size)

    return loader, batch_size

def extract_features(model, loader, feat_dim, batch_sz):
    n_samples = len(loader.dataset)
    camids = None
    ptr = 0
    model_in_pytorch = hasattr(model, "to")

    # Preallocated
    feats = torch.zeros((n_samples, feat_dim), device="cuda")
    labels = torch.zeros(n_samples, dtype=torch.long, device="cuda")

    for imgs, lbls, cams in loader:
        if model_in_pytorch:
            imgs = imgs.to("cuda")
            batch_feats = model(imgs, return_embedding=True)
        else:
            batch_feats = model(imgs, batch_sz)
            batch_feats = torch.from_numpy(batch_feats).to("cuda")

        b = batch_feats.shape[0]

        feats[ptr:ptr+b] = batch_feats
        labels[ptr:ptr+b] = lbls

        del imgs

        ptr += b

    return feats, labels, camids

def load_model(cfg):
    model_filename = cfg.model

    if model_filename.endswith(".pb"):
        from generate_detections import ImageEncoder
        model = ImageEncoder(model_filename)
        return model, model.image_shape
    
    elif model_filename.endswith(".pth"):
        num_classes = int(cfg.num_classes)
        model = MarsSmall128(num_classes=num_classes)
        load_torchWRN_model(model_filename, model)
        torch.set_grad_enabled(False)
        model.to("cuda")
        model.eval()
        return model, (128, 64)


def fixed_features(feats, ids):
    uniq_ids, inverse_indices = torch.unique(ids, return_inverse=True)
    uniq_idx = torch.unique(inverse_indices)
    num_groups = uniq_ids.size(0)

    counts = torch.bincount(inverse_indices).float().unsqueeze(1)

    feats_mean = feats[uniq_idx]
    expanded_means = feats_mean[inverse_indices]
    dot_product = (feats * expanded_means).sum(1)
    row_distances = 1 - dot_product

    sum_dist = torch.zeros(num_groups, device=feats.device, dtype=feats.dtype)
    sum_dist.index_add_(0, inverse_indices, row_distances)

    dists = sum_dist / counts.squeeze()

    sorted_idx = torch.argsort(dists, descending=True)
    uniq_ids = uniq_ids[sorted_idx]
    feats_mean = feats_mean[sorted_idx]
    dists = dists[sorted_idx]

    return uniq_ids, feats_mean, dists

def mean_features_vectorized(feats, ids):
    uniq_ids, inverse_indices = torch.unique(ids, return_inverse=True)
    num_groups = uniq_ids.size(0)

    sum_feats = torch.zeros(num_groups, feats.size(1), device=feats.device, dtype=feats.dtype)
    sum_feats.index_add_(0, inverse_indices, feats)

    counts = torch.bincount(inverse_indices).float().unsqueeze(1)

    feats_mean = sum_feats / counts
    expanded_means = feats_mean[inverse_indices]
    dot_product = (feats * expanded_means).sum(1)
    row_distances = 1 - dot_product

    sum_dist = torch.zeros(num_groups, device=feats.device, dtype=feats.dtype)
    sum_dist.index_add_(0, inverse_indices, row_distances)

    dists = sum_dist / counts.squeeze()

    sorted_idx = torch.argsort(dists, descending=True)
    uniq_ids = uniq_ids[sorted_idx]
    feats_mean = feats_mean[sorted_idx]
    dists = dists[sorted_idx]

    return uniq_ids, feats_mean, dists

def mean_features(feats, ids):

    uniq_ids   = []
    feats_mean = []
    dists      = []
    id_count   = 0

    for j in ids:

        # If this was computed already move on
        if j.item() in uniq_ids:
            continue

        id_j = ids == j
        
        # Filter the features
        feats_j = feats[id_j]
        
        mean_feat = torch.mean(feats_j, 0, True)
        feats_mean.append(mean_feat)

        # Intra-class distance
        dt = 1 - feats_j @ mean_feat.T
        d = torch.mean(dt)
        dists.append(d)

        # Register the id
        uniq_ids.append(j)

        id_count += 1

    feats_mean = torch.cat(feats_mean, dim=0).to("cuda")
    dists      = torch.stack(dists).to("cuda")
    uniq_ids   = torch.stack(uniq_ids).to("cuda")

    sorted_idx = torch.argsort(dists, descending=True)
    uniq_ids   = uniq_ids[sorted_idx]
    feats_mean = feats_mean[sorted_idx]
    dists      = dists[sorted_idx]

    return uniq_ids, feats_mean, dists


def inter_id_distances_vectorized(anchor_feats, anchor_ids):
    dist = 1 - anchor_feats @ anchor_feats.T # Tensor: patches x ids 
    min_dist, min_indices = torch.min(dist, dim=1)

    row_indices = torch.arange(dist.size(0), device=dist.device)
    mask = min_indices != row_indices

    confused_ids = anchor_ids[mask]
    distractor_ids = anchor_ids[min_indices][mask]
    
    confusing_dist = min_dist[mask]
 
    return confused_ids, distractor_ids, confusing_dist

def inter_id_distances(anchor_feats, anchor_ids):
    
    dist = 1 - anchor_feats @ anchor_feats.T # Tensor: patches x ids 
    sorted_idx = torch.argsort(dist)
    confused_ids = []
    distractor_ids = []
    confusing_dist = []

    for i, ai in enumerate(anchor_ids):
        j = sorted_idx[i, 0].item()
        if i != j:
            confused_ids.append(ai)
            distractor_ids.append(anchor_ids[j].item())
            confusing_dist.append(dist[i, j].item())

    return confused_ids, distractor_ids, confusing_dist


def save_intra(path, mapf, ids, dists):
    filename = f"{path}/{mapf}-intra-dist-vec.txt"
    with open(filename, 'w') as fd:
        for i, d in zip(ids, dists):
            fd.write(f"{i} {d:.6f}\n")

def save_inter(path, mapf, c_ids, d_ids, dists):
    filename = f"{path}/{mapf}-inter-dist-vec.txt"
    with open(filename, 'w') as fd:
        for ci, di, dd in zip(c_ids, d_ids, dists):
            fd.write(f"{ci} {di} {dd:.6f}\n")

def mine_hard_ids(cfg):
    model, image_shape = load_model(cfg)
    dataset, bsz = load_dataset(cfg, image_shape[:2])
    feats_dim = 128
    
    print("\nExtracting features...")
    feats, ids, _ = extract_features(model, dataset, feats_dim, bsz)

    print("Computing INTRA id distances...")
    u_ids, feats_mean, dists = mean_features_vectorized(feats, ids)
    #u_ids, feats_mean, dists = fixed_features(feats, ids)

    print("Computing INTER id distances...")
    confused_ids, distractor_ids, confusing_dist =  inter_id_distances_vectorized(feats_mean, u_ids)

    print("storing...")
    u_ids = u_ids.to("cpu").numpy()
    dists = dists.to("cpu").numpy()
    confused_ids = confused_ids.to("cpu").numpy()
    distractor_ids = distractor_ids.to("cpu").numpy()
    confusing_dist = confusing_dist.to("cpu").numpy()

    save_intra(cfg.dataset, cfg.map, u_ids, dists)
    save_inter(cfg.dataset, cfg.map, confused_ids, distractor_ids, confusing_dist)

    
def parse_args():
    parser = argparse.ArgumentParser("ReID feature extractor of selected IDs, this"
                                    "is used to test mAP against the mAP of the"
                                    "multiclass-trained model")
    parser.add_argument("--model",
                        help="Path to frozen inference fraph protobuf.",
                        required=True)

    parser.add_argument("--dataset",
                        help="paths where the query id and gallery ids are listed",
                        required=True)

    parser.add_argument("--num_classes",
                        help="Number of classes, optional, only for the torch version.",
                        default=1)
    parser.add_argument("--batch_sz",
                        help="batch size to evaluate",
                        default=512,
                        type=int)
    parser.add_argument("--map",
                        help="List of identities with the respective path",
                        default="train.txt",
                        type=str)

    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()    
    mine_hard_ids(args)
