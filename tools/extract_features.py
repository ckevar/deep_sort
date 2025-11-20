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
    feats = torch.zeros((n_samples, feat_dim))
    labels = torch.zeros(n_samples, dtype=torch.long)

    for imgs, lbls, cams in loader:
        if model_in_pytorch:
            imgs = imgs.to("cuda")
            batch_feats = model(imgs, return_embedding=True).cpu()
        else:
            batch_feats = model(imgs, batch_sz)
            batch_feats = torch.from_numpy(batch_feats)

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

def mean_features(feats, ids):

    uniq_ids = []
    feats_mean = []
    dists = []
    id_count = 0

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
        uniq_ids.append(j.item())

        id_count += 1

    feats_mean = torch.cat(feats_mean, dim=0)
    dists = torch.stack(dists)
    return uniq_ids, feats_mean, dists

def inter_id_distances(anchor_feats, anchor_ids):
    
    dist = 1 - anchor_feats @ anchor_feats.T # Tensor: patches x ids 
    sorted_idx = torch.argsort(dist)
    complicated_ids = []
    confused_with = []

    for i, ai in enumerate(anchor_ids):
        if i != sorted_idx[i, 0]:
            complicated_ids.append(ai)
            confused_with.append(anchor_ids[sorted_idx[i, 0].item()])

    print(complicated_ids)
    print(confused_with)
    return complicated_ids


def rank_ids(meanf, stdf):
    return None

def compute_metrics(cfg):
    model, image_shape = load_model(cfg)
    dataset, bsz = load_dataset(cfg, image_shape[:2])
    feats_dim = 128
    
    print("\nExtracting features...")
    feats, ids, _ = extract_features(model, dataset, feats_dim, bsz)

    print("Computing INTRA id distances...")
    u_ids, feats_mean, dists = mean_features(feats, ids)
    print("Computing INTER id distances...")
    inter_dist               = inter_id_distances(feats_mean, u_ids)

    ranked_ids = rank_ids(feats_mean, feats_std)
    
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
    cmc, mAP = compute_metrics(args)
