#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset
from IFGSM import ifgsm
import pandas as pd

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # ACE is RGB-only, no need for this param.
    # parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2], help='test mode: 1 = RGB, 2 = RGB-D')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False,
                        help='set to true if your queries are not a smooth video')

    parser.add_argument('--render_pose_error_threshold', type=int, default=20,
                        help='pose error threshold for the visualisation in cm/deg')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_frame_skip', type=int, default=1,
                        help='skip every xth frame for long and dense query sequences')

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test_twenty", # "test" To compute the gradient test_twenty L1_IFGSM
        mode=1,  # Default for ACE, we don't need scene coordinates/RGB-D. 0->1
        image_height=opt.image_resolution,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f'testL2IFGSM_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'posesL2IFGSM_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Store coordinates ################################
    #coord_log= output_dir / f'3D_coordL1.txt'

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Coordinates outputl files ########################
   # test_coord = open(coord_log, 'w',1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = vutil.get_rendering_target_path(
            opt.render_target_path,
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=opt.image_resolution,
        )

        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # coord_list = []
    # List to save the dataset 
    list_ifgsm = []
    # Epsilon values
    epsilon = np.round(np.linspace(start=0, stop=1, num=11),1)
    # Load images
    frames = testset._load_image
    # Counter
    counter = 0
    #grad_list=[]
    # Testing loop.
    testing_start_time = time.time()
# with torch.no_grad():
    for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, gt_sc, filenames in testset_loader:
        batch_start_time = time.time()
        batch_size = image_B1HW.shape[0]
            
        image_B1HW = image_B1HW.to(device, non_blocking=True)
        
        # Activate gradient
        image_B1HW.requires_grad=True

        #print(f'Shape image_B1HW:{image_B1HW.shape}')
        # Predict scene coordinates.
        with autocast(enabled=True):
            scene_coordinates_B3HW = network(image_B1HW)


        gt_sc= gt_sc.to(device)
        
        list_ifgsm = ifgsm(model=network, X= image_B1HW, y_true=gt_sc, 
                           epsilon=epsilon, iterations=20, image_og=frames(counter), dataset=list_ifgsm,
                           counter=counter,L1_Loss = True, mean=0.4, sd=0.25, alpha=1)


        # We need them on the CPU to run RANSAC.
        scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

        counter = counter + 1

 
    dataset = pd.DataFrame(list_ifgsm)
    dataset.to_csv('IFGSML2.csv', index = False)

   
    test_log.close()
    pose_log.close()

