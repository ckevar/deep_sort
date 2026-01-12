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

    path = cfg.dataset
    dataset = ReIDListDataset(path, 
                              f"{path}/{cfg.map}", 
                              transform=trans_qg, 
                              relabel=False)
    loader = DataLoader(dataset, batch_size=cfg.batch_sz)

    return loader

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
    
    elif model_filename.endswith(".pth") or model_filename.endswith(".pth_best"):
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

def mean_features_vectorized(feats, ids, penalized=False):
    uniq_ids, inverse_indices = torch.unique(ids, return_inverse=True)
    num_groups = uniq_ids.size(0)

    # -- Centered Features -- #
    sum_feats = torch.zeros(num_groups, feats.size(1), device=feats.device, dtype=feats.dtype)
    sum_feats.index_add_(0, inverse_indices, feats)

    counts = torch.bincount(inverse_indices).float().unsqueeze(1)

    feats_mean = sum_feats / counts
    if not penalized:
        feats_mean = torch.nn.functional.normalize(feats_mean, p=2, dim=1)
    
    # -- Compute Distances -- #
    expanded_means = feats_mean[inverse_indices]
    dot_product = (feats * expanded_means).sum(1)
    row_distances = 1 - dot_product

    # -- Average, min, max distance -- #
    sum_dist = torch.zeros(num_groups, device=feats.device, dtype=feats.dtype)
    max_dist = torch.empty(num_groups, device=feats.device, dtype=feats.dtype)
    min_dist = torch.empty(num_groups, device=feats.device, dtype=feats.dtype)

    max_dist.fill_(float('-inf'))
    min_dist.fill_(float('inf'))

    sum_dist.index_add_(0, inverse_indices, row_distances)
    max_dist.scatter_reduce_(0, inverse_indices, row_distances, reduce='amax', include_self=False)
    min_dist.scatter_reduce_(0, inverse_indices, row_distances, reduce='amin', include_self=False)

    dists = sum_dist / counts.squeeze()

    # -- Sort IDs by average distances -- #
    sorted_idx = torch.argsort(dists, descending=True)
    uniq_ids = uniq_ids[sorted_idx]
    feats_mean = feats_mean[sorted_idx]
    dists    = dists[sorted_idx]
    min_dist = min_dist[sorted_idx]
    max_dist = max_dist[sorted_idx]

    if penalized:
        feats_mean = torch.nn.functional.normalize(feats_mean, p=2, dim=1)

    return uniq_ids, feats_mean, dists, min_dist, max_dist

def inter_id_distances_vectorized(feats, feat_ids, centroids, centroid_ids):
    if feats.device.type == "cpU": feats = feats.to("cuda")
    if centroids.device.type == "cpu": centroids = centroids.to("cuda")

    dist = 1 - feats @ centroids.T # Tensor: patches x ids 

    centroids = centroids.to("cpu")
    feats = feats.to("cpu")

    min_dist, min_indices = torch.min(dist, dim=1)

    # Centroids
    if centroid_ids.device.type == "cpu": centroid_ids = centroid_ids.to("cuda")
    predicted_closest_ids = centroid_ids[min_indices]
    centroid_ids.to("cpu")

    # Hard Positivevs
    if feat_ids.device.type == "cpu": feat_ids = feat_ids.to("cuda")
    mask = predicted_closest_ids != feat_ids
    confused_img_ids = feat_ids[mask]
    feat_ids = feat_ids.to("cpu")

    patch_row = torch.argwhere(mask).squeeze(1) + 1

    # Hard Negatives
    distractor_ids = predicted_closest_ids[mask]
    confusing_dist = min_dist[mask]
 
    return confused_img_ids, distractor_ids, confusing_dist, patch_row

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


def save_intra(outfile, ids, dists, min_d, max_d, penalized=False):
    
    filename = "{}-intra_dist{}".format(
            outfile, 
            "-penalized.txt" if penalized else ".txt")

    with open(filename, 'w') as fd:
        for i, d, md, MD in zip(ids, dists, min_d, max_d):
            fd.write(f"{i} {d:.6f} {md:.6f} {MD:.6f}\n")

def save_inter(outfile, c_ids, d_ids, dists, rows):
    filename = f"{outfile}-inter_dist.txt"
    with open(filename, 'w') as fd:
        for ci, di, dd, row in zip(c_ids, d_ids, dists, rows):
            fd.write(f"{ci} {di} {dd:.6f} {row}\n")

def mine_hard_ids(cfg):
    model, image_shape = load_model(cfg)
    dataset = load_dataset(cfg, image_shape[:2])
    feats_dim = 128
    
    print("\nExtracting features...")
    feats, ids, _ = extract_features(model, dataset, feats_dim, cfg.batch_sz)

    print("Computing intra id distances...")
    u_ids, feats_mean, dists, min_d, max_d = mean_features_vectorized(
            feats, 
            ids,
            penalized=cfg.penalized)

    u_ids_cpu = u_ids.to("cpu").numpy()
    dists     = dists.to("cpu").numpy()
    min_d     = min_d.to("cpu").numpy()
    max_d     = max_d.to("cpu").numpy()
 
    save_intra(cfg.out_file, u_ids, dists, min_d, max_d, penalized=cfg.penalized)
    del u_ids_cpu, dists, min_d, max_d

    print("Computing inter id distances...")
    confused_ids, distractor_ids, confusing_dist, patch_row =  inter_id_distances_vectorized(
            feats,
            ids,
            feats_mean, 
            u_ids)

    confused_ids = confused_ids.to("cpu").numpy()
    distractor_ids = distractor_ids.to("cpu").numpy()
    confusing_dist = confusing_dist.to("cpu").numpy()
    
    save_inter(cfg.out_file, confused_ids, distractor_ids, confusing_dist, patch_row)

    print(f"Saved in {cfg.out_file}*.txt")

def get_basename(filename_plus_extension):
    fe = filename_plus_extension.split(".")
    ext = fe[1]
    basename = fe[0]
    if basename is None:
        print(f"Filename {basename} seems off. Saving with temporary name 'tmp'")
        basename = "tmp"

    return basename

def parse_args():
    parser = argparse.ArgumentParser(
            "ReID feature extractor of selected IDs, this is used to test mAP "
            "against the mAP of the multiclass-trained model."
            )
    
    parser.add_argument("--model",
                        help="Path to frozen inference fraph protobuf.",
                        required=True)

    parser.add_argument("--dataset",
                        help="paths where the query id and gallery ids are listed",
                        required=True)

    parser.add_argument("--num_classes",
                        help="Number of classes, optional, only for the torch "\
                             "version.",
                        default=1)
    
    parser.add_argument("--batch_sz",
                        help="batch size to evaluate",
                        default=512,
                        type=int)
    
    parser.add_argument("--map",
                        help="List of identities with the respective path",
                        default="train.txt",
                        type=str)

    parser.add_argument("--experiment_name",
                        help="experiment name templated: <words_separated_by_underscore>."\
                             " default name: 'default'",
                        default="default",
                        type=str)

    parser.add_argument("--out_dir",
                        help="output directory, by the default it will save in " \
                             "the same directory where the dataset is.",
                        default=None,
                        type=str)

    parser.add_argument("--penalized",
                        action="store_true",
                        help="the intra distance computed will be the penalized intra distance")

    args = parser.parse_args()

    
    # -- check peanlised -- #
    if args.penalized:
        print("\n[Warning:] the computed intra distance will be the penalized distance.")

    # -- Experiment output -- #
    if args.out_dir is None:
        args.out_dir = args.dataset

    map_file = get_basename(args.map)
    args.out_file = f"{args.out_dir}/{map_file}-{args.experiment_name}"

    return args

if "__main__" == __name__:
    args = parse_args()    
    mine_hard_ids(args)
