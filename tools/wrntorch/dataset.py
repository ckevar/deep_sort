from torch.utils.data import Dataset
from PIL import Image
import os

class ReIDListDataset(Dataset):
    def __init__(self, root_dir, list_path, transform=None, label=None):
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
        if None == label:
            label = {
                pid: idx for idx, pid in enumerate(sorted(set(pid for _, pid, _ in self.samples)))}

        self.__relabel__(label)

    def __relabel__(self, label_map):
      self.label_map = label_map
      new_samples = []

      for img_path, pid, camid in self.samples:
        if pid in label_map:
            new_label = label_map[pid]
            new_samples.append((img_path, new_label, camid))
        # else:
        #    PID not in gallery, skip this sample or handle as you wish
      self.samples = new_samples
      self.classes = list(label_map.keys())

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

