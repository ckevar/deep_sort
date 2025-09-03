import torch
import torch.nn as nn
import torch.nn.functional as F

""" Wide Residual Network
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.same_shape = (in_channels == out_channels and stride == 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if not self.same_shape:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class MarsSmall128(nn.Module):
    def __init__(self, num_classes=None):
        super(MarsSmall128, self).__init__()

        # Input convs
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)   # Output: 128x64x32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # Output: 128x64x32
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        # Output: 64x32x32

        # Residual modules
        self.res1 = ResidualBlock(32, 32, stride=1)   # 64x32x32
        self.res2 = ResidualBlock(32, 32, stride=1)   # 64x32x32
        self.res3 = ResidualBlock(32, 64, stride=2)   # 32x16x64
        self.res4 = ResidualBlock(64, 64, stride=1)   # 32x16x64
        self.res5 = ResidualBlock(64, 128, stride=2)  # 16x8x128
        self.res6 = ResidualBlock(128, 128, stride=1) # 16x8x128

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 128)
        self.bn = nn.BatchNorm1d(128)

        # Optional classifier head
        self.classifier = nn.Linear(128, num_classes) if num_classes is not None else None

    def forward(self, x, return_embedding=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)

        if self.classifier and not return_embedding:
            return self.classifier(x)
        return x
""" Model Utils
"""

def save_torchWRN_checkpoint(model_name, model, optimizer, epoch):
    checkpoint_path = f"{model_name}_model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def load_torchWRN_checkpoint(model_name, model, optimizer):
    filename = f"{model_name}_model.pth"

    if not os.path.isfile(filename):
        print(f"\n    No model found to resume: {filename}.")
        exit(1)

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

def load_torchWRN_model(filename, model):
    if not os.path.isfile(filename):
        print(f"\n    [ERROR:] No model {filename} found to fine tune.")
        exit(1)

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])

# Cell
""" Dataset Class
"""

from torch.utils.data import Dataset
from PIL import Image
import os

class ReIDListDataset(Dataset):
    def __init__(self, root_dir, list_path, transform=None, relabel=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    img_path, pid, cam_id = parts
                else:
                    raise ValueError(f"Invalid line format in {list_path}: {line}")
                self.samples.append((img_path, int(pid), int(cam_id)))

        # Map person IDs to class indices starting from 0
        if relabel:
          self.label_map = {pid: idx for idx, pid in enumerate(sorted(set(pid for _, pid, _ in self.samples)))}
          self.relabel(self.label_map)
          
    
    def relabel(self, label_map):
      self.label_map = label_map
      new_samples = []
      for img_path, pid, camid in self.samples:
        if pid in self.label_map:
            new_label = self.label_map[pid]
            new_samples.append((img_path, new_label, camid))
        else:
            # PID not in gallery, skip this sample or handle as you wish
            continue
      self.samples = new_samples
      self.classes = list(self.label_map.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, camid = self.samples[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, camid


""" Training Utils
"""

from torch.utils.data import Sampler
import random
from collections import defaultdict

class PKSampler(Sampler):
    def __init__(self, data_source, P, K):
        self.data_source = data_source
        self.P = P
        self.K = K

        # Build pid -> indices mapping
        self.index_dict = defaultdict(list)
        for idx, (_, pid, _) in enumerate(data_source):
            self.index_dict[pid].append(idx)

        self.pids = list(self.index_dict.keys())

    def __iter__(self):
        indices = []
        pid_list = self.pids.copy()
        random.shuffle(pid_list)
        for i in range(0, len(pid_list), self.P):
            selected_pids = pid_list[i:i+self.P]
            for pid in selected_pids:
                idxs = self.index_dict[pid]
                if len(idxs) >= self.K:
                    chosen = random.sample(idxs, self.K)
                else:
                    chosen = random.choices(idxs, k=self.K)
                indices.extend(chosen)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

# Cell 

from torchvision import transforms
from torch.utils.data import DataLoader
import yaml

""" Load Configuration
"""

def load_config(config_filename):

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if config['training']['k'] > config['patches_per_id']:
        print("\n    Images per id K:", config['training']['k'], "cannot be greater than total patches per ID", config['patched_per_id'])
        exit()

    return config

"""
"""

# Cell

""" Model Evaluation Utils 
"""

from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
import numpy as np

def extract_features(model, loader):
    model.eval()
    all_feats, all_labels, all_camids = [], [], []
    with torch.no_grad():
        for imgs, labels, camids in loader:
            '''
            if USE_CUDA:
                imgs = imgs.cuda()
            '''
            imgs = imgs.cuda()
            feats = model(imgs, return_embedding=True)  # Extract features
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            all_camids.append(camids)

    feats = torch.cat(all_feats,dim=0)
    labels = torch.cat(all_labels,dim=0)
    camids = torch.cat(all_camids,dim=0)

    return feats, labels, camids

def compute_cmc_map(query_feats, query_ids, query_cams,
                    gallery_feats, gallery_ids, gallery_cams):

    query_feats = query_feats.numpy()
    gallery_feats = gallery_feats.numpy()
    gallery_ids = gallery_ids.cpu().numpy()
    gallery_cams = gallery_cams.cpu().numpy()

    cmc = np.zeros(len(gallery_ids))
    all_AP = []
    valid_queries = 0

    for i in range(len(query_feats)):
        qf = query_feats[i]
        q_id = query_ids[i].item()
        q_cam = query_cams[i].item()

        dists = cdist([qf], gallery_feats, metric='cosine')[0]

        sorted_idx = np.argsort(dists)
        sorted_ids = gallery_ids[sorted_idx]
        matches = (sorted_ids == q_id)

        if matches.sum() == 0:
            print(f"Query {q_id}@{i+1} has no correct matches")
            continue  # No correct matches

        rank = np.where(matches)[0][0]
        cmc[rank:] += 1
        valid_queries += 1

        y_true = matches.astype(int)
        y_score = -dists[sorted_idx]

        ap = average_precision_score(y_true, y_score)
        all_AP.append(ap)

    if 0 == valid_queries:
        raise ValueError("Invalid queries")

    cmc = cmc / valid_queries
    mAP = np.mean(all_AP)

    return cmc, mAP


def sort_dataset_by_pid(dataset):
    dataset.samples.sort(key=lambda x: x[1])

def evaluate_mAP_CMCD(config, model):
    transform_qg = transforms.Compose([
        transforms.Resize(tuple(config['resize'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    root_dir = config["root_dir"]
    gallery_dataset = ReIDListDataset(root_dir, 
                                      f"{root_dir}/{config['gallery']}", 
                                      transform=transform_qg)
    query_dataset = ReIDListDataset(root_dir, 
                                    f"{root_dir}/{config['query']}", 
                                    transform=transform_qg, 
                                    relabel=False)
    query_dataset.relabel(gallery_dataset.label_map)

    query_loader = DataLoader(query_dataset, 
                              batch_size=config['evaluation']['batch_size'])
    gallery_loader = DataLoader(gallery_dataset, 
                                batch_size=config['evaluation']['batch_size'])

    query_feats, query_ids, query_cams = extract_features(model, query_loader)
    gallery_feats, gallery_ids, gallery_cams = extract_features(model, gallery_loader)
    distmat = cdist(query_feats.numpy(), gallery_feats.numpy(), metric='cosine')

    return compute_cmc_map(
      query_feats, query_ids, query_cams,
      gallery_feats, gallery_ids, gallery_cams
    )

# Cell

import os

def save_metrics(filename, epoch, average_loss, mAP, cmc):
    fd = open(filename, 'a')
    fd.write(f"{epoch + 1} {average_loss:.4f} {mAP:.4f} {cmc[0]:.4f} {cmc[4]:.4f} {cmc[9]:.4f}\n")
    fd.close()

def mk_metrics(filename):
    fd = open(filename, "w")
    fd.write("# Epoch Loss mAP Rank-1 Rank-5 Rank-10\n")
    fd.close()


# Cell

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import sys


""" Training
"""

def mk_filenames(config):
    reid_dataset = config["root_dir"].split('/')[-1]
    experiment_name = f"{reid_dataset}_%s_lr{config['training']['lr']}_p{config['training']['p']}_k{config['training']['k']}"

    """
    metrics_filename = f"{reid_dataset}_results_{experiment_name}.dat"
    model_filename = f"{reid_dataset}_model_{experiment_name}"
    """
    metrics_filename = experiment_name % "results"
    metrics_filename = metrics_filename + ".dat"

    model_filename = experiment_name % "model"

    return metrics_filename, model_filename

def init_dataset(config):
    transform = transforms.Compose([
        #transforms.Resize((128, 64)),
        transforms.RandomResizedCrop(tuple(config['resize'])),  # random crop & resize to your input size
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    root_dir = config["root_dir"]
    
    train_dataset = ReIDListDataset(
        root_dir=root_dir,
        list_path=f"{root_dir}/{config['train']}",
        transform=transform
    )
    
    training_batch_size=config['training']['p'] * config['training']['k']
    pk_sampler = PKSampler(train_dataset, 
                           P=config['training']['p'], 
                           K=config['training']['k'])
    train_loader = DataLoader(train_dataset, 
                          batch_size=training_batch_size,
                          sampler=pk_sampler, 
                          num_workers=2)
    return len(train_dataset.classes), train_loader


def init_model_to_train(model_filename, model, optimizer, metrics_filename):
    if 2 == len(sys.argv):
        if "resume" == sys.argv[1]:
            return load_torchWRN_checkpoint(model_filename, model, optimizer)

        elif "load_from_model" == sys.argv[1]:
            load_torchWRN_model(sys.argv[2], model)
            mk_metrics(metrics_filename)
            return 0
        else:
            print("\n   [ERROR:] Command not found.\n")
            exit(1)

    else:
        mk_metrics(metrics_filename)
        return 0



if "__main__" == __name__:

    config = load_config()

    ending_epoch = config['training']['epochs']
    lr = float(config['training']['lr'])

    metrics_filename, model_filename = mk_filenames(config)

    # Init Dataset
    num_classes, train_loader = init_dataset(config)

    # Init training hyper parameters
    """ Create Model
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        model = MarsSmall128(num_classes=num_classes).cuda()
    else:
        model = MarsSmall128(num_classes=num_classes)
    """

    model = MarsSmall128(num_classes=num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(torch.device('cuda'))   # Automatic Mixed Precision
    best_mAP = 0

    start_epoch = init_model_to_train(model_filename, model, optimizer, metrics_filename)

    # Training the model
    for epoch in range(start_epoch, ending_epoch):
        model.train()
        running_loss = 0.0
    
        # Saving checkpoint at epochs multiple of 30
        if 0 == epoch % 30:
            save_torchWRN_checkpoint(model_filename + f"_{epoch:02}epoch", 
                                     model, 
                                     optimizer, 
                                     epoch)
        # Learning Rate Sheduling
        '''
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2
        '''
    
        for images, labels, _ in tqdm(train_loader):
    
            optimizer.zero_grad()

            # In case, CUDA isn't available
            '''
            if USE_CUDA:
                images, labels = images.cuda(), labels.cuda()
            else:
                images, labels = images, labels
            '''
    
            images, labels = images.cuda(), labels.cuda()
    
            # Automatic Mixed Precision (only works in cuda
            with autocast(device_type=torch.device("cuda").type):
                outputs = model(images, return_embedding=False)  # returns logits
                loss = criterion(outputs, labels)
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            running_loss += loss.item()
    
    
        average_loss = running_loss / len(train_loader)

        cmc, mAP = evaluate_mAP_CMCD(config, model)
    
        if mAP > best_mAP:
            save_torchWRN_checkpoint(model_filename + "_best", 
                                     model, 
                                     optimizer, 
                                     epoch)
            best_mAP = mAP
    
        save_metrics(metrics_filename, 
                     epoch, 
                     average_loss, 
                     mAP, 
                     cmc)
        save_torchWRN_checkpoint(model_filename,
                                 model, 
                                 optimizer, 
                                 epoch)
                        
        print(f"Epoch\t\tLoss\tmAP\tRank-1\tRank-5\tRank-10")
        print(f"{epoch + 1}\t\t{average_loss:.4f}\t{mAP:.4f}\t{cmc[0]:.4f}\t{cmc[4]:.4f}\t{cmc[9]:.4f}")

