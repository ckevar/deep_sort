import os
import torch

"""
Torch model file handling
"""

def save_torchWRN_checkpoint(filename, model, optimizer, epoch, scaler, best_mAP):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_mAP': best_mAP,
    }, filename)

def load_torchWRN_checkpoint(filename, model, optimizer, scaler):

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"\n    No model found to resume: {filename}.")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_mAP = checkpoint['best_mAP']

    print(f"Resuming training from epoch {start_epoch}.")
    return start_epoch, best_mAP

def _load_torchWRN_model(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"\n    No model found to resume: {filename}.")

    checkpoint = torch.load(filename)
    return checkpoint['model_state_dict']

def load_torchWRN_model(filename, model):
    model.load_state_dict(_load_torchWRN_model(filename))

def load_torchWRN_finetuning(filename, model, mode):
    checkpoint = _load_torchWRN_model(filename)

    if "finetune" == mode:
        # load everything except the final classifier layer
        pretrained_dict = {k: v for k, v in checkpoint.items() if not k.startswith('classifier.')}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
    elif "finetune_keep" == mode:
        model_dict = checkpoint

    model.load_state_dict(model_dict)


"""
Training configuration handling
"""
import yaml

def load_config(config_filename):

    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)

    if config['training']['k'] > config['patches_per_id']:
        raise ValueError("\n    Images per id K:",
                         config['training']['k'],
                         "cannot be greater than total patches per ID",
                         config['patches_per_id'])

    return config

def save_config(cfg):
    cfg_filename = f"{cfg['experiment_log_dir']}/config.yaml"
    yaml_str = yaml.safe_dump(cfg)

    # Init save storage
    save = cfg.get("save", False)
    if not save: cfg["save"] = save

    with open(cfg_filename, "w") as f:
        f.write(yaml_str)
 

"""
Logs, results and model storage 
"""
def mk_metrics(filename):
    fd = open(filename, "w")
    fd.write("# Epoch Loss mAP Rank-1 Rank-5 Rank-10 lr\n")
    fd.close()

def save_metrics(filename, epoch, average_loss, mAP, cmc, lr):
    fd = open(filename, 'a')
    fd.write(f"{epoch + 1} {average_loss:.4f} {mAP:.4f} {cmc[0]:.4f} {cmc[4]:.4f} {cmc[9]:.4f} {lr:.6f}\n")
    fd.close()

def init_logs(cfg, experiment_name):

    # Make log directories
    log_dir = cfg["log_dir"]
    experiment_path = f"{log_dir}/{experiment_name}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(experiment_path, exist_ok=True)

    cfg["experiment_name"] = experiment_name
    cfg["experiment_log_dir"] = experiment_path

    # Make Filenames
    log_dir = cfg["experiment_log_dir"]

    metrics_filename = f"{log_dir}/results.dat"
    model_filename = f"{log_dir}/model.pth"

    cfg['metrics_filename'] = metrics_filename
    cfg['model_filename'] = model_filename




