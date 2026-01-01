#!/usr/bin/env python

import argparse
import cv2
import numpy as np
from PIL import Image
import os
import simplejson as json
import sys
import yaml
from scipy.spatial.transform import Rotation

sys.path.append("../common/")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import loadimages_inference, loadweights, Draw


def draw_coordinate_system(draw, camera_matrix, dist_coeffs, location, quaternion, axis_length=0.1):
    """
    Draw a 3D coordinate system at the object's centroid.
    
    Args:
        draw: Draw object for rendering
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        location: 3D position of the centroid (x, y, z)
        quaternion: Orientation as quaternion (x, y, z, w)
        axis_length: Length of coordinate axes in meters (default 0.1m = 10cm)
    """
    # Convert quaternion to rotation matrix
    # quaternion format is [x, y, z, w]
    rot = Rotation.from_quat(quaternion)
    rotation_matrix = rot.as_matrix()
    
    # Define axis endpoints in object frame (X, Y, Z axes)
    axes_3d = np.array([
        [axis_length, 0, 0],  # X axis (red)
        [0, axis_length, 0],  # Y axis (green)
        [0, 0, axis_length],  # Z axis (blue)
    ])
    
    # Transform axes to world frame
    axes_world = np.dot(rotation_matrix, axes_3d.T).T + location
    
    # Project centroid and axes endpoints to 2D
    centroid_3d = np.array(location).reshape(3, 1)
    axes_world_3d = axes_world.T
    
    # Project points using camera matrix
    centroid_2d, _ = cv2.projectPoints(centroid_3d.T, np.zeros(3), np.zeros(3), 
                                        camera_matrix, dist_coeffs)
    axes_2d, _ = cv2.projectPoints(axes_world_3d.T, np.zeros(3), np.zeros(3),
                                     camera_matrix, dist_coeffs)
    
    # Extract 2D coordinates
    centroid_2d = tuple(centroid_2d[0][0].astype(int))
    x_axis_2d = tuple(axes_2d[0][0].astype(int))
    y_axis_2d = tuple(axes_2d[1][0].astype(int))
    z_axis_2d = tuple(axes_2d[2][0].astype(int))
    
    # Draw axes with different colors
    draw.draw_line(centroid_2d, x_axis_2d, line_color=(255, 0, 0), line_width=5)  # X axis - Red
    draw.draw_line(centroid_2d, y_axis_2d, line_color=(0, 255, 0), line_width=5)  # Y axis - Green
    draw.draw_line(centroid_2d, z_axis_2d, line_color=(0, 0, 255), line_width=5)  # Z axis - Blue
    
    # Draw centroid point
    #draw.draw_dot(centroid_2d, point_color=(255, 255, 255), point_radius=4)


class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""

    def __init__(
        self,
        config,   # config yaml loaded eg dict
        weight,   # path to weight file
        parallel, # was it trained using DDP
        class_name,
    ):
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        # load network model, create PNP solver
        self.model = ModelData(
            name=class_name,
            net_path=weight,
            parallel=parallel
        )
        self.model.load_net_model()
        print("Model Loaded")

        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(
            class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name])
        )
        self.class_name = class_name

        print("Ctrl-C to stop")

    def image_callback(
        self,
        img,
        camera_info,
        img_name,  # this is the name of the img file to save, it needs the .png at the end
        output_folder,  # folder where to put the output
        weight,
        debug=False
    ):
        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(
                camera_info["projection_matrix"]["data"], dtype="float64"
            ).copy()
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype="float64")
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype="float64")
            dist_coeffs.resize((len(camera_info.D), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(
                img, (int(scaling_factor * width), int(scaling_factor * height))
            )

        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        # dictionary for the final output
        dict_out = {"camera_data": {}, "objects": []}

        # Detect object
        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect,
            grid_belief_debug=debug
        )

        # Publish pose and overlay cube on image
        for _, result in enumerate(results):
            if result["location"] is None:
                continue

            loc = result["location"]
            ori = result["quaternion"]

            dict_out["objects"].append(
                {
                    "class": self.class_name,
                    "location": np.array(loc).tolist(),
                    "quaternion_xyzw": np.array(ori).tolist(),
                    "projected_cuboid": np.array(result["projected_points"]).tolist(),
                }
            )
            # Draw the cube
            if None not in result["projected_points"]:
                points2d = []
                for pair in result["projected_points"]:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_color)

            # Draw the coordinate system at the centroid instead of the cube
            draw_coordinate_system(draw, camera_matrix, dist_coeffs, loc, ori, axis_length=10)

        # create directory to save image if it does not exist
        img_name_base = img_name.split("/")[-1]
        output_path = os.path.join(
            output_folder,
            weight.split("/")[-1].replace(".pth", ""),
            *img_name.split("/")[:-1],
        )
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        im.save(os.path.join(output_path, img_name_base))
        if belief_imgs is not None:
            belief_imgs.save(os.path.join(output_path, "belief_maps.png"))

        json_path = os.path.join(
            output_path, ".".join(img_name_base.split(".")[:-1]) + ".json"
        )
        # save the json files
        with open(json_path, "w") as fp:
            json.dump(dict_out, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outf",
        default="output",
        help="Where to store the output images and inference results.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file (.mp4) to process.",
    )
    parser.add_argument(
        "--config",
        default="../config/config_pose.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--camera",
        default="../config/camera_info.yaml",
        help="Path to camera info file",
    )

    parser.add_argument(
        "--weights",
        "--weight",
        "-w",
        required=True,
        help="Path to weights or folder containing weights. If path is to a folder, then script "
        "will run inference with all of the weights in the folder. This could take a while if "
        "the set of test images is large.",
    )

    parser.add_argument(
        "--parallel",
        action='store_true',
        help="Were the weights trained using DDP; if set to true, the names of later weights "
        " will be altered during load to match the model"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output video (default: 30).",
    )

    parser.add_argument(
        "--object",
        required=True,
        help="Name of class to run detections on.",
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help="Generates debugging information, including raw belief maps and annotation of "
        "the results"
    )

    opt = parser.parse_args()

    # load the configs
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(opt.outf, exist_ok=True)

    # Load model weights
    weights = loadweights(opt.weights)

    if len(weights) < 1:
        print(
            "No weights found at specified directory. Please check --weights flag and try again."
        )
        exit()
    else:
        print(f"Found {len(weights)} weights. ")

    # Check if video file exists
    if not os.path.exists(opt.video):
        print(f"Video file not found: {opt.video}")
        exit()

    # Open input video
    video_capture = cv2.VideoCapture(opt.video)
    if not video_capture.isOpened():
        print(f"Error opening video file: {opt.video}")
        exit()

    # Get video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {opt.video}")
    print(f"Total frames: {total_frames}")
    print(f"Original FPS: {original_fps}")
    print(f"Resolution: {video_width}x{video_height}")

    for w_i, weight in enumerate(weights):
        print(f"\nProcessing with weight {w_i + 1} of {len(weights)}: {weight}")
        dope_node = DopeNode(config, weight, opt.parallel, opt.object)

        # Construct output path for this weight
        output_path = os.path.join(
            opt.outf,
            weight.split("/")[-1].replace(".pth", "")
        )
        os.makedirs(output_path, exist_ok=True)

        # Reset video to beginning
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Prepare to read first frame to get processed dimensions
        ret, first_frame = video_capture.read()
        if not ret:
            print("Error reading first frame")
            continue

        # Convert BGR to RGB for processing
        first_frame_rgb = first_frame[..., ::-1].copy()
        
        # Check if downscaling will be applied
        height, width, _ = first_frame_rgb.shape
        scaling_factor = float(dope_node.downscale_height) / height
        if scaling_factor < 1.0:
            output_width = int(scaling_factor * width)
            output_height = int(scaling_factor * height)
        else:
            output_width = width
            output_height = height

        # Create video writer for output
        video_path = os.path.join(output_path, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, opt.fps, (output_width, output_height))

        # Reset video to beginning for actual processing
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        last_results = None  # Store last detection results
        last_camera_matrix = None
        last_dist_coeffs = None
        process_interval = 4  # Process every 4 frames
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            
            # Convert BGR to RGB
            frame_rgb = frame[..., ::-1].copy()

            # Check if we should run inference on this frame
            should_process = (frame_count - 1) % process_interval == 0
            
            if should_process:
                print(f"Processing frame {frame_count} of {total_frames} (running inference)...", end='\r')
                
                # Process frame through DOPE
                # Update camera matrix and distortion coefficients
                if dope_node.input_is_rectified:
                    P = np.matrix(
                        camera_info["projection_matrix"]["data"], dtype="float64"
                    ).copy()
                    P.resize((3, 4))
                    camera_matrix = P[:, :3]
                    dist_coeffs = np.zeros((4, 1))
                else:
                    camera_matrix = np.matrix(camera_info.K, dtype="float64")
                    camera_matrix.resize((3, 3))
                    dist_coeffs = np.matrix(camera_info.D, dtype="float64")
                    dist_coeffs.resize((len(camera_info.D), 1))

                # Downscale frame if necessary
                height, width, _ = frame_rgb.shape
                scaling_factor = float(dope_node.downscale_height) / height
                if scaling_factor < 1.0:
                    camera_matrix[:2] *= scaling_factor
                    frame_rgb = cv2.resize(
                        frame_rgb, (int(scaling_factor * width), int(scaling_factor * height))
                    )

                dope_node.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
                dope_node.pnp_solver.set_dist_coeffs(dist_coeffs)

                # Detect object
                results, _ = ObjectDetector.detect_object_in_image(
                    dope_node.model.net, dope_node.pnp_solver, frame_rgb, dope_node.config_detect,
                    grid_belief_debug=False
                )
                
                # Store results for next frames
                last_results = results
                last_camera_matrix = camera_matrix
                last_dist_coeffs = dist_coeffs
            else:
                print(f"Processing frame {frame_count} of {total_frames} (reusing previous results)...", end='\r')
                
                # Use stored results from last processed frame
                results = last_results
                camera_matrix = last_camera_matrix
                dist_coeffs = last_dist_coeffs
                
                # Still need to downscale frame for consistent output size
                height, width, _ = frame_rgb.shape
                scaling_factor = float(dope_node.downscale_height) / height
                if scaling_factor < 1.0:
                    frame_rgb = cv2.resize(
                        frame_rgb, (int(scaling_factor * width), int(scaling_factor * height))
                    )

            # Copy and draw image
            img_copy = frame_rgb.copy()
            im = Image.fromarray(img_copy)
            draw = Draw(im)

            # Draw results on frame (whether new or reused)
            if results is not None:
                for _, result in enumerate(results):
                    if result["location"] is None:
                        continue

                    loc = result["location"]
                    ori = result["quaternion"]

                    # Draw the cube
                    if None not in result["projected_points"]:
                        points2d = []
                        for pair in result["projected_points"]:
                            points2d.append(tuple(pair))
                        draw.draw_cube(points2d, dope_node.draw_color)

                    # Draw the coordinate system at the centroid
                    draw_coordinate_system(draw, camera_matrix, dist_coeffs, loc, ori, axis_length=10)

            # Convert PIL image back to OpenCV format (RGB to BGR)
            output_frame = np.array(im)
            output_frame_bgr = output_frame[..., ::-1].copy()

            # Write frame to output video
            video_writer.write(output_frame_bgr)

        print(f"\nProcessed {frame_count} frames")
        
        # Release video writer
        video_writer.release()
        print(f"Output video saved to: {video_path}")
        print("------")

    # Release video capture
    video_capture.release()
    print("\nDone!")