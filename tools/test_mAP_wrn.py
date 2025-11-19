import argparse
from generate_detections import ImageEncoder
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
        #arr = np.array(img, dtype=np.uint8)
        arr = np.array(img)
        arr = arr[:, :, ::-1].copy() # from RGB to BGR
        return arr

def load_dataset(cfg, image_shape):
    path = cfg.dataset
    batch_size = cfg.batch_sz

    model_file =  cfg.model

    if model_file.endswith(".pb"):
        transform_qg = transforms.Compose([
            transforms.Resize(image_shape),
            ToNumpyUint8(),
        ])
    elif model_file.endswith(".pth"):
        transform_qg = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:x*255),
            #transforms.Lambda(lambda x:x[[1, 2, 0], ...]) # RGB -> BGR
        ])


    gallery_dataset = ReIDListDataset(path,
                                      f"{path}/gallery.txt",
                                      transform=transform_qg)
    query_dataset = ReIDListDataset(path,
                                    f"{path}/query.txt",
                                    transform=transform_qg,
                                    relabel=False)
    query_dataset.relabel(gallery_dataset.label_map)

    query_loader = DataLoader(query_dataset, batch_size=batch_size)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size)

    return query_loader, gallery_loader, batch_size

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

def compute_cmc_map_in_gpu(query_feats,
                           query_ids,
                           gallery_feats,
                           gallery_ids,
                           batch_size=32092):
    query_feats = query_feats.to("cuda")
    query_ids = query_ids.to("cuda")
    gallery_ids = gallery_ids.to("cuda")

    len_gallery_feats = len(gallery_feats)

    # Preallocation

    dist       = torch.empty(len_gallery_feats, device="cuda", dtype=gallery_feats.dtype)
    sorted_idx = torch.empty(len_gallery_feats, device="cuda", dtype=torch.int64)
    cmc        = torch.zeros(len(gallery_ids) , device="cuda", dtype=torch.int64)
    all_AP     = torch.tensor(0.0, device="cuda")
    valid_queries = torch.tensor(0, device="cuda", dtype=torch.int64)

    for i in range(len(query_feats)):
        queryf = query_feats[i:i+1]

        # Cosine distance
        for j in range(0, len_gallery_feats, batch_size):
            galleryf = gallery_feats[j:j+batch_size].to("cuda")
            num_moved = galleryf.shape[0]
            sim = queryf @ galleryf.T
            dist[j:j+num_moved] = (1 - sim).squeeze(0)

        # mAP and Ranks
        sorted_idx = torch.argsort(dist)
        sorted_ids = gallery_ids[sorted_idx]

        q_id = query_ids[i].item()
        y_true = (sorted_ids == q_id)
        tp = torch.cumsum(y_true, dim=0)
        total_positives = tp[-1] # total sum

        if 0 == total_positives:
            print(f"Query {q_id}@{i + 1} has no correct matches.")
            continue

        # Ranks
        rank = torch.where(y_true)[0][0]
        cmc[rank:] += 1

        # AP
        precision = tp / (torch.arange(len(y_true), device=y_true.device, dtype=torch.float32) + 1)
        ap = (precision * y_true).sum() / total_positives

        all_AP += ap
        valid_queries += 1


    if 0 == valid_queries.item():
        raise ValueError("Invalid queries, it's highly likely that the gallery doesn't contain any query's id.")

    all_AP = all_AP / valid_queries
    cmc = cmc / valid_queries


    cmc = cmc.cpu().numpy()
    mAP = all_AP.cpu()

    return cmc, mAP

import random

def load_model(cfg):
    model_filename = cfg.model

    if model_filename.endswith(".pb"):
        model = ImageEncoder(model_filename)
        return model, model.image_shape
    
    elif model_filename.endswith(".pth"):
        num_classes = int(cfg.num_classes)
        model = MarsSmall128(num_classes=num_classes)
        load_torchWRN_model(model_filename, model)
        torch.set_grad_enabled(False)
        model.to("cuda")
        model.eval()
        return model, (64, 128)

def compute_metrics(cfg):
    
    model, image_shape = load_model(cfg)
    query, gallery, bsz = load_dataset(cfg, image_shape[:2])

    feats_dim = 128

    Q_feats, Q_ids, Q_cams = extract_features(model, query, feats_dim, bsz)
    G_feats, G_ids, G_cams = extract_features(model, gallery, feats_dim, bsz)
    
    return compute_cmc_map_in_gpu(
            Q_feats, Q_ids,
            G_feats, G_ids,
            batch_size=512000)

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
                        default=512)

    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()    
    cmc, mAP = compute_metrics(args)
    print(f"mAP: {mAP}, Rank-1: {cmc[0]}, Rank-5: {cmc[4]}, Rank-9:{cmc[9]}")
