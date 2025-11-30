import sys

if sys.path[0].endswith("/tools"):
    from freeze_torch_model import MarsSmall128
    from freeze_torch_model import load_torchWRN_model
    from generate_detections import extract_image_patch
else:
    from .freeze_torch_model import MarsSmall128
    from .freeze_torch_model import load_torchWRN_model
    from .generate_detections import extract_image_patch

import numpy as np
import torch
from torchvision import transforms

def __run_in_batches__(model, x, data_len, batch_size):
    num_batches = int(data_len / batch_size)
    all_feats = []
    
    s, e = 0, 0
    with torch.no_grad():
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data = x[s:e,:,:,:]
            batch_data = batch_data.cuda()
            feats = model(batch_data, return_embedding=True)
            all_feats.append(feats.cpu())

        if e < data_len:
            batch_data = x[e:, :, :, :]
            batch_data = batch_data.cuda()
            print(x.shape)
            exit(1)
            feats = model(batch_data, return_embedding=True)
            all_feats.append(feats.cpu())

    feats = torch.cat(all_feats, dim=0)

    return feats
    

class ImageTorchEnconder(object):
    def __init__(self, checkpoint_filename, num_classes=388):

        # Transformation Required for the
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            ])

        # Design Hyper Parameters
        self.feature_dim = 128 # WRN design in the deepsort paper
        self.image_shape = [128, 64, 3] # WRN design in the deepsort paper
        self.model = MarsSmall128(num_classes=num_classes)
        load_torchWRN_model(checkpoint_filename, self.model)
        self.model.to("cuda")
        self.model = self.model.eval()
       
    def __call__(self, data_x, batch_size=1):
        data_len = data_x.shape[0]
        if 0 == data_len:
            return torch.empty((0, self.feature_dim), 
                               dtype=torch.float32)
        img = torch.stack([self.transform(imgx) for imgx in data_x])

        feats = __run_in_batches__(self.model, img, data_len, batch_size)
        return feats


def create_box_encoder(model_filename, num_classes, batch_size=32):
    image_encoder = ImageTorchEnconder(model_filename, num_classes)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("\n  [WARNING:] Failed to extract image patch: %s.\n" % str(box))
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            patch = patch[:, :, ::-1]
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)
    
    return encoder



