
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

if sys.path[0].endswith("/tools"):
    from wrntorch.utils import (
        save_torchWRN_checkpoint, 
        load_torchWRN_checkpoint, 
        load_torchWRN_model, 
        load_torchWRN_finetuning,
        load_config,
        save_config,
        init_logs,
        mk_metrics,
        save_metrics
        )

    from wrntorch.dataset import ReIDListDataset
else:
    from .wrntorch.utils import (
        save_torchWRN_checkpoint, 
        load_torchWRN_checkpoint, 
        load_torchWRN_model, 
        load_torchWRN_finetuning,
        load_config,
        save_config,
        init_logs,
        mk_metrics,
        save_metrics
        )

    from .wrntorch.dataset import ReIDListDataset


"""# Model """

class Conv2Same(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=0, bias=bias
            )
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        stride = self.stride
        K = self.kernel_size

        # Compute SAME padding
        out_h = (H + stride - 1) // stride
        out_w = (W + stride - 1) // stride

        pad_h = max(0, (out_h - 1) * stride + K - H)
        pad_w = max(0, (out_w - 1) * stride + K - W)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left 

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_first=False):
        super(ResidualBlock, self).__init__()
        same_shape = (in_channels == out_channels and stride == 1)

        self.pre_bn = None
        if not is_first:
            self.pre_bn = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.999)

        self.conv1 = Conv2Same(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=stride,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)

        #self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        self.conv2 = Conv2Same(out_channels, out_channels, 3, stride=1, bias=True)

        self.downsample = None
        if not same_shape:
            #self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.downsample = Conv2Same(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, _in):
        x = _in

        if self.pre_bn:
            x = self.pre_bn(x)
            x = self.elu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.dropout(x)

        x = self.conv2(x)

        if self.downsample:
            _in = self.downsample(_in)
        
        x += _in
        return x

class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = torch.matmul(features, weight.t())
        s = F.softplus(self.scale)
        logits = s * (cosine + 0.1)
        return logits

class MarsSmall128(nn.Module):
    def __init__(self, num_classes=None):
        super(MarsSmall128, self).__init__()

        # Input convs
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)   # Output: 128x64x32
        self.conv1 = Conv2Same(3, 32, kernel_size=3, stride=1, bias=True)   # Output: 128x64x32
        self.conv1_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.999) 
        #self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)  # Output: 128x64x32
        self.conv2 = Conv2Same(32, 32, kernel_size=3, stride=1, bias=True)  # Output: 128x64x32
        self.conv2_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.999)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)        # Output: 64x32x32
        
        self.elu = nn.ELU(inplace=True)

        # Residual modules
        self.res1 = ResidualBlock(32, 32, stride=1, is_first=True)   # 64x32x32
        self.res2 = ResidualBlock(32, 32, stride=1)   # 64x32x32
        self.res3 = ResidualBlock(32, 64, stride=2)   # 32x16x64
        self.res4 = ResidualBlock(64, 64, stride=1)   # 32x16x64
        self.res5 = ResidualBlock(64, 128, stride=2)  # 16x8x128
        self.res6 = ResidualBlock(128, 128, stride=1) # 16x8x128

        # Final layers
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(16384, 128)
        self.bn = nn.BatchNorm1d(128, eps=1e-3, momentum=0.999)

        # Optional classifier head: print of failure #FAIL #Fail #failure #fail
        # I have ran many experiments and im not sure which model or when a 
        # certain is using a linear classifier or a softmax classifier, like the
        # one for the deepSORT paper.

        self.classifier = CosineClassifier(128, num_classes)
        #self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.elu(self.conv1_bn(self.conv1(x)))
        x = self.elu(self.conv2_bn(self.conv2(x)))
        x = self.pool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.elu(x)

        x = F.normalize(x, p=2, dim=1)

        if self.classifier and not return_embedding:
            return x, self.classifier(x)
        return x


"""# Evaluation Utils """

def extract_features(model, loader, feat_dim, num_cams=1):
    model.eval()
    n_samples = len(loader.dataset)

    # Preallocate
    feats = torch.zeros((n_samples, feat_dim))
    labels = torch.zeros(n_samples, dtype=torch.long)

    camids = None
    if num_cams > 1:
        camids = torch.zeros(n_samples, dtype=torch.long)

    ptr = 0
    with torch.no_grad():
        for imgs, lbls, cams in loader:
            imgs = imgs.cuda()
            batch_feats = model(imgs, return_embedding=True).cpu()
            b = batch_feats.size(0)

            feats[ptr:ptr+b] = batch_feats
            labels[ptr:ptr+b] = lbls
            if num_cams > 1: camids[ptr:ptr+b] = cams

            ptr += b

    return feats, labels, camids

from sklearn.metrics import average_precision_score
import numpy as np

def legacy_compute_cmc_map(query_feats, query_ids, query_cams,
                    gallery_feats, gallery_ids, gallery_cams):

    query_feats = query_feats.numpy()
    gallery_feats = gallery_feats.numpy()
    gallery_ids = gallery_ids.cpu().numpy()

    if gallery_cams is not None:
        gallery_cams = gallery_cams.cpu().numpy()

    cmc = np.zeros(len(gallery_ids))
    all_AP = 0.0
    valid_queries = 0

    for i in range(len(query_feats)):
        qf = query_feats[i]
        q_id = query_ids[i].item()
        # TODO: Uncomment if using more than one camera
        #q_cam = query_cams[i].item()

        # Compute cosine distance
        dists = 1 - qf @ gallery_feats.T

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
        all_AP += ap

    if 0 == valid_queries:
        raise ValueError("Invalid queries")

    cmc = cmc / valid_queries
    mAP = all_AP / valid_queries

    return cmc, mAP

def compute_cmc_map_in_gpu(query_feats,
                           query_ids,
                           query_cams,
                           gallery_feats,
                           gallery_ids,
                           gallery_cams,
                           batch_size=32092):

    # Legacy: gallery_ids = gallery_ids.cpu().numpy() # Leave at GPU
    query_feats = query_feats.to("cuda")
    query_ids = query_ids.to("cuda")
    gallery_ids = gallery_ids.to("cuda")

    if gallery_cams is not None:
        gallery_cams = gallery_cams.cpu().numpy()

    len_gallery_feats = len(gallery_feats)

    dists_gpu = torch.empty(len_gallery_feats, device="cuda", dtype=gallery_feats.dtype)
    sorted_idx_gpu = torch.empty(len_gallery_feats, device="cuda", dtype=torch.int64)
    cmc_gpu = torch.zeros(len(gallery_ids), device="cuda", dtype=torch.int64)
    all_AP_gpu = torch.tensor(0.0, device="cuda")  # float32 by default
    valid_queries_gpu = torch.tensor(0, device="cuda", dtype=torch.int64)

    for i in range(len(query_feats)):
        queryf = query_feats[i:i+1]

        # Cosine Distance in GPU
        for j in range(0, len_gallery_feats, batch_size):
            galleryf = gallery_feats[j:j+batch_size].to("cuda")
            num_moved = galleryf.shape[0]
            sim = queryf @ galleryf.T
            dists_gpu[j:j+num_moved] = (1 - sim).squeeze(0)

        # Compute mAP and Ranks
        sorted_idx_gpu = torch.argsort(dists_gpu)
        sorted_ids = gallery_ids[sorted_idx_gpu]

        q_id = query_ids[i].item()

        y_true_gpu = (sorted_ids == q_id)
        tp = torch.cumsum(y_true_gpu, dim=0)
        total_positives = tp[-1] # Total sum

        if total_positives == 0:
            print(f"Query {q_id}@{i+1} has no correct matches")
            continue  # No correct matches

        # Ranks
        rank_gpu = torch.where(y_true_gpu)[0][0]
        cmc_gpu[rank_gpu:] += 1

        # AP
        precision = tp / (torch.arange(len(y_true_gpu), device=y_true_gpu.device, dtype=torch.float32) + 1)
        ap_gpu = (precision * y_true_gpu).sum() / total_positives
        all_AP_gpu = all_AP_gpu + ap_gpu

        valid_queries_gpu += 1

    if 0 == valid_queries_gpu.item():
        raise ValueError("Invalid queries. It's highly likely that the gallery doesn't contain any query's id")

    all_AP_gpu = all_AP_gpu / valid_queries_gpu
    cmc_gpu = cmc_gpu / valid_queries_gpu

    cmc = cmc_gpu.cpu().numpy()
    mAP  = all_AP_gpu.cpu()

    return cmc, mAP

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch.optim as optim
from torch.amp import GradScaler, autocast
import random
from collections import defaultdict

def evaluate_mAP_CMCD(config, model, feat_dims):
    transform_qg = transforms.Compose([
        transforms.Resize(tuple(config['resize'])),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


    root_dir = config["root_dir"]

    gallery_dataset = ReIDListDataset(root_dir,
                                      f"{root_dir}/{config['gallery']}",
                                      transform=transform_qg)

    query_dataset = ReIDListDataset(root_dir,
                                    f"{root_dir}/{config['query']}",
                                    transform=transform_qg,
                                    label=gallery_dataset.label_map)

    query_loader = DataLoader(query_dataset,
                              batch_size=config['evaluation']['batch_size'])
    gallery_loader = DataLoader(gallery_dataset,
                                batch_size=config['evaluation']['batch_size'])

    query_feats, query_ids, query_cams = extract_features(model, query_loader, feat_dims)
    gallery_feats, gallery_ids, gallery_cams = extract_features(model, gallery_loader, feat_dims)

    return compute_cmc_map_in_gpu(
      query_feats, query_ids, query_cams,
      gallery_feats, gallery_ids, gallery_cams,
      batch_size=512000
    )

"""# Training Utils"""

"""## training utils | Freeze/Unfreeze Functions """
def __set_requires_grad(modules:list, unfreeze:bool):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = unfreeze

        if unfreeze:
            m.train()
        else:
            m.eval()

def freeze_model(model, phase):

    if phase >= 2: # Freeze just shallow layers
        __set_requires_grad([model.conv1, model.conv1_bn,
                             model.conv2, model.conv2_bn],
                            False)

    if phase >= 3:
        __set_requires_grad([model.pool, model.res1], False)
  
    if phase >= 4: # Freeze half feature descriptor
        __set_requires_grad([model.res2, model.res3], False)
  
    if phase >= 5: # Freeze just before the last feature map
        __set_requires_grad([model.res4, model.res5], False)
  
    if phase >= 6: # freeze the entire backbone
        __set_requires_grad([model.res6], False) # Last feature map
  
    if 7 == phase:
        __set_requires_grad([model.fc, model.bn], False)

def unfreeze_backbone(model, phase):

    print("unfreezing", phase)

    if 6 == phase:
        __set_requires_grad([model.conv1, model.conv1_bn,
                             model.conv2, model.conv2_bn],
                            True)

    if phase == 5:
        __set_requires_grad([model.pool, model.res1], True)

    if phase == 4:
        __set_requires_grad([model.res2, model.res3], True)

    if phase == 3:
        __set_requires_grad([model.res4, model.res5], True)

    if phase == 2:
        __set_requires_grad([model.res6], True)
    
    if phase == 1:
        __set_requires_grad([model.fc, model.bn], True)

"""## training utils | Initialisers """
def init_LearningRate(cfg):

    # Initial learning rate o reference learning rate
    lr = float(cfg['training']['lr'])

    # Coefficients of the learning rate
    lr_scheduling = cfg["training"].get("lr_scheduling", None)
    if lr_scheduling is None:
        cfg['training']['lr_scheduling'] = None

    elif isinstance(lr_scheduling, list):
        lr_scheduling = [float(l) for l in lr_scheduling]

    elif isinstance(lr_scheduling, str):
        if "None" == lr_scheduling:
            return lr, None, None

        lr_scheduling = [float(lr_scheduling)]

    # when to change the learning rate
    lr_schedule_at = cfg['training'].get("lr_schedule_at", None)
    if lr_schedule_at is None:
        cfg['training']['lr_schedule_at'] = None

    elif isinstance(lr_schedule_at, list):
        if len(lr_schedule_at) != len(lr_scheduling):
            raise ValueError("lr scheduling and lr_schedule_have to be the same length.")
    else:
        if lr_scheduling is None:
            raise ValueError("new learning rate or lr dividend is missing.")
        lr_schedule_at = [int(lr_schedule_at)]

    return lr, lr_scheduling, lr_schedule_at

def init_dataset(config):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(tuple(config['resize'])),  # random crop & resize to your input size
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    pre_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    post_transform = transforms.Compose([
        transforms.RandomErasing(p=0.5)
    ])

    pre_transform_at = config["training"].get("pre_transform_at", None)
    if pre_transform_at is None: 
        config["training"]["pre_transform_at"] = pre_transform_at
        pre_transform = None

    post_transform_at = config["training"].get("post_transform_at", None)
    if post_transform_at is None: 
        config["training"]["post_transform_at"] = post_transform_at
        post_transform = None

    root_dir = config["root_dir"]

    train_dataset = ReIDListDataset(
        root_dir=root_dir,
        list_path=f"{root_dir}/{config['train']}",
        transform=transform,
        pre_transform=pre_transform,
        post_transform=post_transform,
    )

    train_batch_sz = config['training']['p'] * config['training']['k']
    pk_sampler = PKSampler(train_dataset,
                           P=config['training']['p'],
                           K=config['training']['k'])
    
    #KAGGLE torch.set_float32_matmul_precision('high')
    #KAGGLE torch.backends.cuda.matmul.allow_tf32 = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_sz,
        sampler=pk_sampler,
        num_workers=2
    )

    config["num_classes"] = len(train_dataset.classes)

    return config["num_classes"], train_loader, train_dataset

def init_model_to_train(config, mode, model, optimizer, scaler):
    if "resume" == mode:
        return load_torchWRN_checkpoint(config['model_filename'], model, optimizer, scaler)

    elif "from_model" == mode:
        load_torchWRN_model(config['model_filename'], model)
        mk_metrics(config['metrics_filename'])
        return 0, 0

    elif "train" == mode:
        mk_metrics(config['metrics_filename'])
        return 0, 0

    elif "finetune" == mode or "finetune_keep" == mode:
        load_torchWRN_finetuning(config['finetune']['model'], model, mode)
        mk_metrics(config['metrics_filename'])
        freeze_model(model, config['finetune']['phase'])
        model.eval()
        return 0, 0

    raise ValueError(f"\n    Unknown mode: {mode}. They can be `train` (from scratch), `resume` (model under the metrics dir), `from_model` (finetune)")

def fix_seed(seed, determinism_level):
    """
    seed: seed to place in every random module
    determinism_level: there are three levels of determinsm:
    - 0: no determisn at all, everything is random, so it means seed will be ignored.
    - 1: only placing the seed
    - 2: placing seed plus making torch deterministic
    """

    if determinism_level >= 1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Upon multiple GPUs

    if 2 == determinism_level:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def init_seed(cfg):
    seed = cfg['training'].get('seed', -1)
    if seed < 0:
        seed = random.randint(0, 2**32 - 1) 

    print("seed:", seed)

    determinism_level = cfg['training'].get("determinism_level", 0)
    if 0 == determinism_level:
        cfg["training"]["determinism_level"] = determinism_level

    fix_seed(seed, determinism_level)


"""## training utils | Batch Sampler """

class PKSampler(Sampler):
    def __init__(self, data_source, P, K):
        self.data_source = data_source
        self.P = P
        self.K = K

        # Build pid -> indices mapping
        self.index_dict = defaultdict(list)
        for idx, (_, pid, _) in enumerate(data_source.samples):
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


"""## training utils | Losses """

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels):
        """
        Feats need to be l2-normalised.
        """
        dist = 1 - (feats @ feats.T)

        N = feats.size(0)
        loss = 0.0
        triplets = 0

        for i in range(N):
            anchor_label = labels[i]
            pos_mask = labels == anchor_label
            neg_mask = labels != anchor_label

            pos_dists = dist[i][pos_mask]
            neg_dists = dist[i][neg_mask]

            # prevent nans
            if neg_dists.numel() == 0:
                continue

            # Remove the anchor itself (distance = 0)
            pos_dists = pos_dists[pos_dists > 1e-6] 

            # Check if there are any *other* positives
            if pos_dists.numel() == 0:
                continue

            # Find HARDEST pairs, not easiest
            hardest_pos = pos_dists.max() # Farthest positive
            hardest_neg = neg_dists.min() # Closest negative

            # max(0, hardest_pos - hardest_neg + margin), makes zero once it reaches the margin
            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            # Soft Margin
            #triplet_loss = F.softplus(hardest_pos - hardest_neg)

            # Only add to the loss if the triplet is "active"
            if triplet_loss > 0:
                loss += triplet_loss
                triplets += 1

        if triplets == 0:
            return torch.tensor(0.0, requires_grad=True, device=feats.device)

        return loss / triplets

class Criterion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mode = cfg["criterion"]

        self.mode = None
        self.epochs = None
        self.criterion = [None, None]
        self.alpha = cfg.get("alpha", 1.0) # Cross Entropy Weight

        if mode in ("crossentropy", "combined"):
            # There was no need to add label_smoothing for MOT17
            self.criterion[0] = nn.CrossEntropyLoss(label_smoothing=0.1) 
            self.mode = 0

        if mode in ("tripletloss", "combined"):
            self.criterion[1] = TripletLoss(margin=cfg["triplet_margin"])
            self.mode = 1

        if "combined" == mode:
            self.mode = 2
            self.beta = cfg.get("beta", 1.0)

        if self.mode is None:
            raise ValueError("`criterion` is required in configuration file, `tripletloss`, `crossentropy`, `both`.")

    def forward(self, feats, logits, labels, criterion_epoch):
        """ 
        # Disables cross entropy at:
        if 57 == epoch:
            self.mode = 1
        # Disables Triplet loss
        if criterion_epoch == 27:
            self.mode = 0
        """

        if 0 == self.mode: # cross entropy alone
            return self.alpha * self.criterion[0](logits, labels)

        elif 1 == self.mode: # Triplet Loss alone
            return self.criterion[1](feats, labels)

        elif 2 == self.mode: # Cross entropy + Triplet Loss
            """
            # Sheduling Coefficients
            #if 56 == epoch: self.alpha, self.beta = 1.0, 6.6 # This is going to be our last hope.
            #if 56 == epoch: self.mode = 1 # triplet loss only
            """
            lossCE = self.criterion[0](logits, labels)
            lossTP = self.criterion[1](feats, labels)
            loss = self.alpha * lossCE + self.beta * lossTP
            return loss

        raise ValueError("Something went wrong during loss calculation. Make sure the criterios are correctly set in the configuration.")

"""## training utils | Schedulers """
def attempt_unfreeze_backbone(model, optimizer, curr_epoch, cfg):

    if curr_epoch not in cfg["uepoch"]:
        return

    for i, e in enumerate(cfg["uepoch"]):
        if e != curr_epoch:
            continue
        
        unfreeze_backbone(model, cfg["uphase"][i])

        if cfg["ulr"] is None:
            continue

        if "None" == cfg["ulr"][i]:
            continue

        for param_group in optimizer.param_groups:
            param_group["lr"] *= cfg["ulr"][i]
    
def attempt_update_lr(model, opt, epoch, lr_scheduling, lr_schedule_at):

    if epoch not in lr_schedule_at:
        return

    idx = lr_schedule_at.index(epoch)
    lr_coeff = lr_scheduling[idx]

    for param_group in opt.param_groups:
        param_group['lr'] *= lr_coeff

def attempt_update_dataAugmentation(dataset, epoch, cfg):
    if epoch == cfg["pre_transform_at"]:
        dataset.enable_pre_transform(True)
        print("Enabling pre transforms")
    if epoch == cfg["post_transform_at"]:
        dataset.enable_post_transform(True)
        print("Enabling post transforms")

"""## training utils | Storage Utils """

def save_checkpoint(cfg, model, optimizer, epoch, scaler, best_mAP, mAP):
    fname = cfg['model_filename']

    # Save the best model based on mAP
    if mAP > best_mAP:
        best_mAP = mAP
        save_torchWRN_checkpoint(f"{fname}_best", model, optimizer, epoch, scaler, best_mAP)

    if not cfg["save"]:
        return

    # Save the model in the latest training epoch
    save_torchWRN_checkpoint(fname, model, optimizer, epoch, scaler, best_mAP)

    # Save the model periodically
    if epoch % cfg["training"]["checkpoint_period"]:
        return
    save_torchWRN_checkpoint(f"{fname}_{epoch:03}epoch", model, optimizer, epoch, scaler, best_mAP)
    
"""# Train """
from tqdm import tqdm
def train(config_file, mode="train", experiment_name="default"):
    config = load_config(config_file)

    # Seed and Learning Rate
    init_seed(config)
    lr, lr_scheduling, lr_schedule_at = init_LearningRate(config)

    # Directories, Filenames Model and Metrics
    init_logs(config, experiment_name)

    # Init Dataset
    num_classes, train_loader, tdataset = init_dataset(config)

    # Save new config yaml
    save_config(config)

    # Init training hyper parameters
    model = MarsSmall128(num_classes=num_classes).cuda()

    criterion = Criterion(config["training"])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8) # 1e-8 by design
    scaler = GradScaler(torch.device('cuda'))   # Automatic Mixed Precision

    start_epoch, best_mAP = init_model_to_train(config, mode, model, optimizer, scaler)
    ending_epoch = config['training']['epochs']

    # Summary
    print(f"Metrics @ {config['metrics_filename']}, Model @ {config['model_filename']}")
    print(f"Number of classes: {num_classes} | Important upon deployment.")

    # Training the model
    for epoch in range(start_epoch, ending_epoch):

        torch.set_grad_enabled(True)
        model.train()
        running_loss = 0.0

        # Schedulers {
        # - Unfreezing the backbone
        if "finetune" == mode:
            attempt_unfreeze_backbone(model, optimizer, epoch, config["unfreeze_backbone"])

        # - Learning Rate
        if lr_schedule_at is not None: 
            attempt_update_lr(model, optimizer, epoch, lr_scheduling, lr_schedule_at)

        # - Post/Pre Data Augmentation
        attempt_update_dataAugmentation(tdataset, epoch, config["training"])
        # }
        train_sz = 0
        for images, labels, _ in tqdm(train_loader):

            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()

            # Automatic Mixed Precision (only works in cuda
            with autocast(device_type=torch.device("cuda").type):
                feats, logits = model(images, return_embedding=False)  # returns Embeddings
                loss = criterion(feats, logits, labels, epoch)


            if 0.0 == loss.item():
                print("[Warning:] Loss got 0.0")
                continue

            train_sz += 1
            scaler.scale(loss).backward()
            
            # Avoids nans
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        del images, labels, feats, logits
        torch.cuda.empty_cache()

        average_loss = running_loss / train_sz
        print(f"Average loss: {average_loss}.")

        cmc, mAP = evaluate_mAP_CMCD(config, model, 128)
        lr = optimizer.param_groups[0]['lr']

        save_checkpoint(config, model, optimizer, epoch, scaler, best_mAP, mAP)
        save_metrics(config['metrics_filename'], epoch, average_loss, mAP, cmc, lr)

        print(f"Epoch\t\tLoss\tmAP\tRank-1\tRank-5\tRank-10")
        print(f"{epoch + 1}\t\t{average_loss:.4f}\t{mAP:.4f}\t{cmc[0]:.4f}\t{cmc[4]:.4f}\t{cmc[9]:.4f}\t{lr:.10f}")

import argparse
def user_config():
    parser = argparse.ArgumentParser("WRN Trainer")

    parser.add_argument("--mode",
                       help="how to train, from scratch `train`, `finetune`, `resume`",
                       type=str,
                       default="train")
    parser.add_argument("--cfg",
                       help="configuration file",
                       default=None)
    parser.add_argument("--experiment_name",
                        help="This will create a directory where results and models will be saved.",
                        type=str,
                        default="default")

    return parser.parse_args()

if "__main__" == __name__:

    args = user_config()
    train(args.cfg, mode=args.mode, experiment_name=args.experiment_name)

