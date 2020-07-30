import os
import time
import argparse
import json
import sys
sys.path.append('/home/j0118570/Documents/lamp_status_191111/')

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pdb

from general_tools import object_visualizer

def parse_arguments():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Read one json file and convert it.')
    parser.add_argument(
        "--input_path", "-in",
        default="/home/j0118570/Documents/KPIT/annotation_results/sample_real_world_cuboid/Input-1/",
        help="path to annotation_results folder",
    )
    parser.add_argument("--mode",
        type=str,
        help="which result to visualize",
        choices=["lamp", "lane", "3d"],
        default="3d")
    return parser.parse_args()

class KpitLampStatus(object):
    def __init__(self, obj_face_side, tail_lamp_status, turn_indicator_status, vertices_2d):
        # input from raw kpit annotation
        self.obj_face_side = obj_face_side
        self.tail_lamp_status = tail_lamp_status
        self.turn_indicator_status = turn_indicator_status
        self.vertices_2d = vertices_2d

    def get_all_status(self):
        tv_direction = self.get_direction_status()
        tv_brake_on = self.get_brake_status()
        tv_left_turn_on = self.get_left_turn_status()
        tv_right_turn_on = self.get_right_turn_status()
        tv_hazard_on = self.get_hazard_status()
        return tv_direction, tv_brake_on, tv_left_turn_on, tv_right_turn_on, tv_hazard_on

    def get_direction_status(self):
        # input:  kpit definetion -> ['REAR', 'FRONT', 'RIGHT_SIDE', 'LEFT_SIDE']
        # output: vbp definetion -> front=0, back=1
        if self.obj_face_side in ['REAR', 'RIGHT_SIDE', 'LEFT_SIDE']:
            tv_direction = 1
            return tv_direction
        elif self.obj_face_side == 'FRONT':
            tv_direction = 0
            return tv_direction
        else:
            raise KeyError("This obj_face_side is not defined: {}".format(self.obj_face_side))

    def get_brake_status(self):
        # input:  kpit definetion -> (0=off, 1=on, 2=unknown for ['left', 'right', 'up'])
        # output: vbp definetion -> brake_on = True or False
        left_light_status = self.tail_lamp_status['left']
        right_light_status = self.tail_lamp_status['right']
        up_light_status = self.tail_lamp_status['up']
        brake_on = False

        if up_light_status == 1:
            brake_on = True
        elif up_light_status == 0 or up_light_status == 2:
            if left_light_status==1 and right_light_status==1:
                brake_on = True
        return brake_on

    def get_left_turn_status(self):
        # input:  kpit definetion -> (0=off, 1=on, 2=unknown for ['left', 'right')
        # output: vbp definetion -> left_on = True or False
        left_winker_status = self.turn_indicator_status['left']
        right_winker_status = self.turn_indicator_status['right']
        left_turn_on = False

        if left_winker_status == 1 and right_winker_status == 0:
            left_turn_on = True
        return left_turn_on

    def get_right_turn_status(self):
        # input:  kpit definetion -> (0=off, 1=on, 2=unknown for ['left', 'right')
        # output: vbp definetion -> left_on = True or False
        left_winker_status = self.turn_indicator_status['left']
        right_winker_status = self.turn_indicator_status['right']
        right_turn_on = False

        if right_winker_status == 1 and left_winker_status == 0:
            right_turn_on = True
        return right_turn_on

    def get_hazard_status(self):
        # input:  kpit definetion -> (0=off, 1=on, 2=unknown for ['left', 'right')
        # output: vbp definetion -> left_on = True or False
        left_winker_status = self.turn_indicator_status['left']
        right_winker_status = self.turn_indicator_status['right']
        hazard_on = False

        if right_winker_status == 1 and left_winker_status == 1:
            hazard_on = True
        return hazard_on

    def get_2d_box(self):
    # '''
    # Calculate a fake 2D bbox from the annotated cuboid vertices on image
    # :param vertices_2d: {"x1": , "y1": , "x2": , "y2": ,..., "x4": ,"y4": }
    # :return: [x_min, y_min, x_max, y_max]
    # '''
    # //todo: case that coordinate exceed image frame
        vertices_2d = self.vertices_2d
        x_list = [vertices_2d['x1'], vertices_2d['x2'], vertices_2d['x3'], vertices_2d['x4']]
        y_list = [vertices_2d['y1'], vertices_2d['y2'], vertices_2d['y3'], vertices_2d['y4']]
        x_min = Decimal(min(x_list)).quantize(Decimal('0.00'))
        x_max = Decimal(max(x_list)).quantize(Decimal('0.00'))
        y_min = Decimal(min(y_list)).quantize(Decimal('0.00'))
        y_max = Decimal(max(y_list)).quantize(Decimal('0.00'))
        box_2d = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)
        return box_2d

def draw_polygon(ax, lane_class, polygon_coords, num):
    # lane_classes = {'Free_Space': 'b', 'Lane': 'g', 'Road_Structure': 'r', 'Traffic_signal': 'y'}
    # if lane_class in lane_classes:
    #     colors = lane_classes[lane_class]
    # else:
    #     colors = "k"

    plt.style.use('fivethirtyeight')
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    length = len(polygon_coords)
    x_list = np.zeros(length)
    y_list = np.zeros(length)
    for i, polygon_coord in enumerate(polygon_coords):
        x_list[i] = polygon_coord[0]
        y_list[i] = polygon_coord[1]

    # T = np.arctan2(y_list, x_list)
    # ax.scatter(x_list, y_list, s=75, c=colors, label=lane_class)
    ax.plot(x_list, y_list, linewidth=1, marker='o', label=lane_class, markersize=2)

def get_corners(dimensions, location, rotation_z):
    #dimensions = np.clip(dimensions, a_min=1.5, a_max=5) # Removed because the actual dimensions were getting modified

    # R matrix formula modified because the rotation is happening across Z-axis
    R = np.array([[np.cos(rotation_z), -np.sin(rotation_z), 0],
                  [np.sin(rotation_z), np.cos(rotation_z), 0],
                  [0, 0, 1]], dtype=np.float32)

    h, w, l = dimensions
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2] # Forward direction (length)
    y_corners = [-w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2] # Horizontal direction (width)
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] # vertical direction (height)

    corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
    corners_3D += location.reshape((3, 1))
    return corners_3D

def draw_projection(corners, P2, ax, color):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    orders = [[0, 1, 2, 3, 0],
              [4, 5, 6, 7, 4],
              [2, 6], [3, 7],
              [1, 5], [0, 4]]
    for order in orders:
        ax.plot(projection[0, order], projection[1, order],
                color=color, linewidth=2)
    return

def main():
    args = parse_arguments()
    work_dir = os.getcwd()
    video_base_path = args.input_path               # ./Dataset-x/Input-x/.
    veh_painter = object_visualizer.VehiclePainter()

    if os.path.isdir(video_base_path):
        scenario_folder_list = os.listdir(video_base_path)
        for scenario_folder_name in scenario_folder_list:
            scenario_folder_name = os.path.join(work_dir, video_base_path, scenario_folder_name)
            for i in os.listdir(scenario_folder_name):
                if os.path.splitext(i)[1] == '.json':
                    img_mark = os.path.splitext(i)[0]
                    # store label file path for 1 scenario
                    label_file_path = os.path.join(scenario_folder_name, i)
                else:
                    continue

            with open(label_file_path, 'r') as label:
                label_per_scenario = json.load(label)
                print("{0} imgs in file {1}".format(len(label_per_scenario), label_file_path))
                # print("Converting labels now ...")

                for label_per_img in tqdm(label_per_scenario):      # Items in Scenario-X folder
                    img_name = label_per_img['image_name']
                    img_path = os.path.join(scenario_folder_name, 'Input_Frames', img_name)
                    img_orig = Image.open(img_path).convert('RGB')
                    img_orig_w, img_orig_h = img_orig.size
                    print('image_path: ', img_path)

                    if args.mode == "lamp":
                        # draw bbox and lamp_status on img_orig for each vehicle in this image
                        for idx in label_per_img['3d_info']:
                            tv_id = idx['id']
                            tv_type = idx['class']
                            if tv_type == 'PEDESTRIAN' or tv_type == 'BICYCLE':
                                continue

                            # Light status dict
                            kls = KpitLampStatus(idx['obj_face_side'], idx['tail_lamp_status'], idx['turn_indicator_status'], idx['box_coords'])
                            tv_direction, tv_brake_on, tv_left_turn_on, tv_right_turn_on, tv_hazard_on = kls.get_all_status()
                            light_status = dict(brake_on=tv_brake_on,left_turn_on=tv_left_turn_on,right_turn_on=tv_right_turn_on,
                                                hazard_turn_on=tv_hazard_on,direction=tv_direction)

                            # This painter receive img_orig and bbox_one_vehicle each time, and draw bbox on img_orig recursively
                            print('tv_id: ', tv_id, 'tv_left_turn_on: ', tv_left_turn_on, 'tv_right_turn_on: ', tv_right_turn_on)
                            tv_2d_box = kls.get_2d_box()
                            veh_painter.draw_mmdet_results_det_single(img_orig, tv_2d_box, light_status)

                        # show image
                        img_pil_2_cv = cv2.cvtColor(np.asarray(img_orig), cv2.COLOR_RGB2BGR)
                        cv2.imshow("Vehicle Light Status", img_pil_2_cv)
                        k = cv2.waitKey(0)
                        if k == ord('q'):
                            continue

                    elif args.mode == "lane":

                        #### Show original image ####
                        # fig = plt.figure(figsize=(20, 10))
                        # ax = fig.gca()
                        # ax.grid(False)
                        # ax.set_axis_off()
                        # ax.set_xlim((0, 3840))
                        # ax.set_ylim((1920, 0))
                        # ax.imshow(img_orig)
                        # plt.show()

                        #### Show embeded image ####
                        fig = plt.figure(figsize=(20, 10))
                        ax = fig.gca()
                        ax.grid(False)
                        ax.set_axis_off()
                        ax.set_xlim((0, 3840))
                        ax.set_ylim((1920, 0))
                        ax.imshow(img_orig)

                        for i, idx in enumerate(label_per_img['semantic_info']):
                            lane_class = idx['class']
                            lane_polygon = idx['polygon_coords']
                            draw_polygon(ax, lane_class, lane_polygon, i)

                        # plt.savefig('./visualize/{}_proj'.format(index))
                        # plt.close()
                        plt.legend(loc = 'upper right')
                        plt.show()

                    elif args.mode == '3d':
                        intrinsic_mtx = np.asarray([[6274.71132, 0.0, 1920.0], [0.0, 6258.53001, 960.0], [0.0, 0.0, 1.0]]) # Camera matrix
                        extrinsic_mtx = np.asarray([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.3], [1.0, 0.0, 0.0, 0.0]]) # Rotation matrix | Translation vector
                        P2 = intrinsic_mtx.dot(extrinsic_mtx) # projection matrix calculated from intrinsic and extrinsic parameters

                        # Prepare the original image on ax in order to draw 3d-box on it
                        fig = plt.figure(figsize=(38, 19))
                        ax = fig.gca()
                        ax.grid(False)
                        ax.set_axis_off()
                        ax.imshow(img_orig) 

                        for data in label_per_img['3d_info']:
                            if data['cuboid_type'] == '2D_cuboid':
                                continue

                            real_plane_coords = data['real_plane_coords']
                            dimensions = [real_plane_coords['height'], real_plane_coords['width'], real_plane_coords['length']]
                            # Corrected the sequence and changed "real_plane_coords[centroidY]" to positive value
                            location = np.asarray([real_plane_coords['centroidX'], real_plane_coords['centroidY'], real_plane_coords['centroidZ']])

                            # real_plane_coords['rot1'] is the yaw change and its rotating axis is Z-axis
                            rotation_z = real_plane_coords['rot1']

                            corners = get_corners(dimensions, location, rotation_z)

                            draw_projection(corners, P2, ax, color=(0, 1, 0)) #P2 - projection matrix

                        # plt.savefig(output_path + obj_data['image_name'])
                        # plt.close()
                        plt.show()

                    
    else:
        raise IOError("Annatation results path not exsit!")

if __name__ == '__main__':
    main()

