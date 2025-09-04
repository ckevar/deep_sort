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

import time


def gather_sequence_info(sequence_dir, data_type, detector_file, feature_extractor_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detector_file : str
        Path to the detector file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detector: A detector object.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    if "MOT17" == data_type:
        image_dir = os.path.join(sequence_dir, "img1")
    elif "KITTI" == data_type:
        image_dir = sequence_dir

    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    print("image_filenames keys")
    for key in image_filenames.keys():
        print(key, image_filenames[key].split('/')[-1])
        break


    # Load Detector
    if "yolo" in detector_file:
        from ultralytics import YOLO
        detector = YOLO(detector_file)
    else:
        print(f"\n  Detector {detector_file} not supported yet.\n")
        exit(1)

    # Load feature extractor 
    if feature_extractor_file.endswith("pb"):
        from tools.generate_detections import create_box_encoder
        feature_extractor = create_box_encoder(feature_extractor_file, batch_size=32)

    elif feature_extractor_file.endswith("pth"):
        from tools.generate_torch_detections import create_box_encoder
        if "mot17" in feature_extractor_file:
            train_ids = 388
        elif "kitti" in feature_extractor_file:
            train_ids = 450
        elif "waymo" in feature_extractor_file:
            train_ids = 20982
        else:
            print(f"\n  Model {feature_extractor_file} not supported, unknown train_ids.\n")
            exit(1)
        feature_extractor = create_box_encoder(feature_extractor_file, train_ids, batch_size=32)
    else:
        print(f"\n  Model {feature_extractor_file} not supported.\n")
        exit(1)

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    """

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())
    print(f"\n min frame idx: {min_frame_idx}\n max_frame_idx: {max_frame_idx}\n")
    print(f"\n min_frame_idx name: {image_filenames[min_frame_idx]}\n")
    """
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detector": detector,
        "image_size": image_size,
        #"min_frame_idx": min_frame_idx,
        #"max_frame_idx": max_frame_idx,
        "feature_extractor": feature_extractor,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(dets, confs, feats):
    detection_list = []

    for bbox, conf, feat in zip(dets, confs, feats):
        detection_list.append(Detection(bbox, conf, feat))

    return detection_list


def unravel_detections_confs(detections, min_height):
    confs = detections.boxes.conf.cpu().numpy()
    detections = detections.boxes.xyxy.cpu().numpy()
    detections[:, 2:4] = detections[:, 2:4] - detections[:, 0:2]
    detections = detections[detections[:, 3] > min_height,:]
    return detections, confs

total_et = 0
total_frame = 0

def run(sequence_dir, data_type, detector_file, feature_extractor_file, 
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
    detector_file : str
        Path to the detections file.
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

    seq_info = gather_sequence_info(sequence_dir, data_type, detector_file, feature_extractor_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    total_et = 0
    total_frame = 0

    def frame_callback(vis, frame_idx):
        global total_et
        global total_frame
        print("Processing frame %05d" % frame_idx)

        print(f"\n frame name: {seq_info['image_filenames'][frame_idx]}")

        frame = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        start_time = time.time()

        detections = seq_info["detector"](frame, verbose=False)[0]
        detections, confs = unravel_detections_confs(detections, min_detection_height)

        feats = seq_info["feature_extractor"](frame, detections)

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

def store_results(output_file, results, data_type):

    if "MOT17" == data_type:
        save_format = "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
    elif "KITTI" == data_type:
        save_format = "%d %d pedestrian 0 0 -10 %.2f %.2f %.2f %.2f -10 -10 -10 -1000 -1000 -1000 -10"
        results = np.array(results)
        results[:, 4:] += results[:, 2:4]
    else:
        print("\n Data type {data_type} not supported.")
        exit(1)

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
        "--experiment_name", help="Experiment name. It will create a folder with "
        "results.", default=None, required=True)
    parser.add_argument(
        "--load_feature_extractor", help="Feature extractor model file",
        default=None, required=True)
    parser.add_argument(
        "--data_dir", help="Dataset directory, so far it supports, MOT17, "
        "KITTI and WaymoV2-MOT format", default=None, required=True)
    parser.add_argument(
        "--data_type", help="It supports MOT and KITTI formats",
        default=None, required=True)
    # }
    return parser.parse_args()

def mk_output_dir(root_dir, results_dir):
    dirs = root_dir.split('/')
    output_dir = '/'.join(dirs[:-1]) + f"/results/{results_dir}"
    try:
        os.mkdir(output_dir)
    except:
        pass
    return output_dir

if __name__ == "__main__":
    args = parse_args()

    sequences = os.listdir(args.data_dir)
    output_dir = mk_output_dir(args.data_dir, args.experiment_name)

    for seq_dir in sequences:
        output_file = f"{output_dir}/{seq_dir}.txt"
        sequence_dir = f"{args.data_dir}/{seq_dir}"
        
        run(sequence_dir, args.data_type, args.load_detector, args.load_feature_extractor, 
            output_file, args.min_confidence, args.nms_max_overlap, 
            args.min_detection_height, args.max_cosine_distance, args.nn_budget,
            args.display)



