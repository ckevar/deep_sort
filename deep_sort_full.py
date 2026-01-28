# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from sequenceloader import SequenceLoader

import time


def load_detector(detector_file):
    if "yolo" in detector_file:
        from ultralytics import YOLO
        return YOLO(detector_file)
    else:
        raise ValueError(f"Detector {detector_file} not supported.\n")

def load_feature_extractor(fe_file, fe_cfg=None, fe_type="wrn"):
    if fe_type == "wrn":
        if fe_file.endswith("pb"):
            from tools.generate_detections import create_box_encoder
            return create_box_encoder(fe_file, batch_size=32)

        elif fe_file.endswith("pth"):
            from tools.generate_torch_detections import create_box_encoder
            if "mot17" in fe_file: train_ids = 388
            elif "kitti" in fe_file: train_ids = 451
            elif "waymo" in fe_file: train_ids = 20982
            else: raise ValueError(f"Model {fe_file} not supported, unknow train_ids.\n")
            return create_box_encoder(fe_file, train_ids, batch_size=128)

        else:
            raise ValueError(f"Mode {fe_file} not supported.\n")

    elif fe_type == "fastreid":
        if fe_cfg is None:
            raise ValueError("FastReID models need a *.yaml configuration file.\n")
        
        import sys
        sys.path.append("trackers/BoostTrack")
        sys.path.append("trackers/BoostTrack/tracker")
        sys.path.append("trackers/BoostTrack/external")

        from tracker.embedding import EmbeddingComputer
        from default_settings import GeneralSettings

        embedder = EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True)
        embedder.load(fe_file, fe_cfg)

        return embedder

    else:
        raise ValueError(f"Feature extractor type {fe_type} not supported.\n")

 


def gather_sequence_info(sequence_dir, data_type):
    """Gather sequence information, such as image filenames, detections,

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    data_type: str
        Dataset format MOT17 or KITTI.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * image_size: Image size (height, width).
        * update_ms: frame rate of the sequence.

    """

    image_size = None
    update_ms = None
    if "MOT17" == data_type:
        image_dir = os.path.join(sequence_dir, "img1")
    elif "KITTI" == data_type:
        image_dir = sequence_dir
    else:
        raise ValueError(f"--data_type only supports MOT17 or KITTI.\n")

    # Image filenames
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    # Image size
    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape

    # Sequence Frame Rate (FPS)
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "image_size": image_size,
        "update_ms": update_ms
    }
    return seq_info

import torch
def create_detections(dets, confs, feats):
    detection_list = []

    for bbox, conf, feat in zip(dets, confs, feats):
        detection_list.append(Detection(bbox, conf, feat))

    return detection_list


def unwrap_detections_ltwh_confs(detections, min_height):
    confs = detections.boxes.conf.unsqueeze(1)
    boxes = detections.boxes.xyxy.clone().detach()
    boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, :2]

    out = torch.cat([boxes, confs], dim=1)
    out = out[out[:, 3] > min_height, :]
    out = out.cpu().numpy()

    return out[:, :4], out[:, 4]

total_et = 0
total_frame = 0

def run(sequence_dir, data_type, detector, feature_extractor, 
        output_file, min_confidence, nms_max_overlap, 
        min_detection_height, max_cosine_distance, nn_budget, display):

    print(f"\n  Processing Sequence: {sequence_dir}.\n")

    global total_et
    global total_frame

    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    data_type : str
        Dataset format, MOT17 or KITTI.
    detector : object
        Detector Model.
    feature_extractor: 
        Feature Extractor Model.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maximum suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, data_type)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    tracker = Tracker(metric)
    results = []
    total_et = 0
    total_frame = 0

    def frame_callback(vis, frame_idx):
        global total_et
        global total_frame
        #print("Processing frame %05d" % frame_idx)

        frame = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        start_time = time.time()

        detections = detector(frame, verbose=False)[0]
        detections, confs = unwrap_detections_ltwh_confs(detections, min_detection_height)
        
        if hasattr(feature_extractor, "compute_embedding"):
            boxes = detections.copy()
            boxes[:, 2:4] += boxes[:, :2]
            feats = feature_extractor.compute_embedding(frame, boxes, f"{sequence_dir}:{frame_idx}")
        else:
            feats = feature_extractor(frame, detections)

        # Load image and generate detections.
        detections = create_detections(detections, confs, feats)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maximum suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            vis.set_image(frame.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        total_et += time.time() - start_time
        total_frame += 1

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    print(f"Time elapsed: {total_et}, FPS: {total_frame/total_et}\n")

    # Store results.
    store_results(output_file, results, data_type)

    if display:
        cv2.destroyAllWindows()

def store_results(output_file, results, data_type):

    if "MOT17" == data_type:
        save_format = "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
    elif "KITTI" == data_type:
        save_format = "%d %d pedestrian 0 0 -10 %.2f %.2f %.2f %.2f -10 -10 -10 -1000 -1000 -1000 -10"
        results = np.array(results)
        results[:, 4:6] += results[:, 2:4]
    else:
        raise ValueError(f"Dataset format --data_type={data_type} not supported.\n")

    f = open(output_file, 'w')
    for row in results:
        print(save_format % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    f.close()


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--load_detector", help="Path to detector (supports yolo only).", 
        default=None, required=True)
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)


    # New {
    parser.add_argument(
        "--experiment_name", help="Results directory (not path)."
        "results.", default=None, required=True)
    parser.add_argument(
        "--load_feature_extractor", help="Feature extractor model file",
        default=None, required=True)
    parser.add_argument(
        "--feature_extractor_cfg",
        help="Path to the FastReID config file.",
        type=str,
        default=None)
    parser.add_argument(
        "--feature_extractor_type",
        type=str,
        help="This could be 'wrn' or 'fastreid'",
        default="wrn")


    parser.add_argument(
        "--data_dir", help="Dataset directory, so far it supports, MOT17, "
        "KITTI and WaymoV2-MOT format", default=None, required=True)
    parser.add_argument(
        "--data_type", help="It supports MOT and KITTI formats",
        default="MOT")
    parser.add_argument(
        "--overwrite", type=bool, help="If True, it processes the entire dataset from scratch. If false, it resumes from the cache file, if there exists a cache file.",
        default=False)
    # }

    return parser.parse_args()

def mk_output_dir(root_dir, results_dir):
    if '/' == root_dir[-1]:
        root_dir = root_dir[:-1]

    dirs = root_dir.split('/')
    output_dir = '/'.join(dirs[:-1]) + f"/results/{results_dir}"
    try:
        os.mkdir(output_dir)
    except:
        pass
    return output_dir

class Cache(object):
    def __init__(self, experiment_name, overwrite=False):
        self.cache_filename = '.cache_' + experiment_name
        if not os.path.isfile(self.cache_filename) or overwrite:
            fd = open(self.cache_filename, 'w')
            fd.close()

    def processed(self, seq_dir):
        cached_seq = False
        with open(self.cache_filename, 'r') as fd:
            cache_list = fd.read()
            cached_seq = seq_dir in cache_list

        return cached_seq 

    def ack(self, seq_dir):
        with open(self.cache_filename, 'a') as fd:
            fd.write(f"{seq_dir}\n")

if __name__ == "__main__":

    args = parse_args()

    sequences = os.listdir(args.data_dir)
    output_dir = mk_output_dir(args.data_dir, args.experiment_name)
    cache = Cache(args.experiment_name, args.overwrite)
    detector = load_detector(args.load_detector)
    feature_extractor = load_feature_extractor(args.load_feature_extractor, 
                                               args.feature_extractor_cfg,
                                               args.feature_extractor_type)

    for seq_dir in sequences:

        if cache.processed(seq_dir):
            continue

        output_file = f"{output_dir}/{seq_dir}.txt"
        sequence_dir = f"{args.data_dir}/{seq_dir}"

        run(sequence_dir, args.data_type, detector, feature_extractor, 
            output_file, args.min_confidence, args.nms_max_overlap, 
            args.min_detection_height, args.max_cosine_distance, args.nn_budget,
            args.display)

        cache.ack(seq_dir)


