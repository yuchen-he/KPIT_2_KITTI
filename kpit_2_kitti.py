# KPIT annotation --> KITTI 3D annotation
# Format: [type, truncated, occluded, alpha, 2D_bbox(4), dimensions(3), location(3), rotation_y, score]
# Reference: https://blog.csdn.net/hit1524468/article/details/79766805

from __future__ import print_function
import argparse
import json
import os
import pdb

import numpy as np
from decimal import Decimal
from tqdm import tqdm
import cv2


def parse_arguments():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Read one json file and convert it.')
    parser.add_argument(
        "--input_path", "-in",
        default="./annotation_results/Dataset-1A/Input-1/",
        help="path to annotation_results folder",
    )
    parser.add_argument(
        "--save_path", "-s",
        default="./converted_labels/Dataset-1A/Input-1/",
        help="path to save converted labels in .txt",
    )
    parser.add_argument(
        "--resize_image", "-r",
        default="True",
        help="if or not to resize images and save them",
    )
    return parser.parse_args()

def resize_image(imgs, image_path, save_path):
    if os.path.isdir(save_path):
        for img in imgs:
            img_path_full = os.path.join(image_path, img)
            img_orig = cv2.imread(img_path_full)
            # (3840, 1920) -> (1280, 640) -> (1248, 384)
            img_resized = cv2.resize(img_orig, (1280, 640), interpolation=cv2.INTER_CUBIC)
            img_crop = img_resized[128:512, 16:1264, :]

            # Convert img name to be %6d
            img = img.zfill(10)
            cv2.imwrite(os.path.join(save_path, img), img_crop)
    else:
        raise IOError('This path for saving resized images is not exist: {}'.format(save_path))

def convert_type(kpit_type):
    kpit_list = ['CAR', 'BUS', 'TRUCK', 'MOTORBIKE', 'PEDESTRIAN']
    if kpit_type == kpit_list[0]:
        kitti_type = 'Car'
    elif kpit_type == kpit_list[1]:
        kitti_type = 'Van'
    elif kpit_type == kpit_list[2]:
        kitti_type = 'Truck'
    elif kpit_type == kpit_list[3]:
        kitti_type = 'DontCare'
    elif kpit_type == kpit_list[4]:
        kitti_type = 'Pedestrian'
    else:
        raise KeyError("KPIT_vehicle_type is not defined!")
    return kitti_type

def convert_occl_label(kpit_occl):
    kpit_label = ['NOT_OCCLUDED', 'OCCLUDED']
    if kpit_occl == kpit_label[0]:
        kitti_occl = 0
    elif kpit_occl == kpit_label[1]:
        kitti_occl = 1
    else:
        raise KeyError("KPIT_Occlusion_label is not defined!")
    return kitti_occl

def cal_2d_box(vertices_2d):
    '''
    Calculate a fake 2D bbox from the annotated cuboid vertices on image
    :param vertices_2d: {"x1": , "y1": , "x2": , "y2": ,..., "x4": ,"y4": }
    :return: [x_min, y_min, x_max, y_max]
    '''
    x_list = [vertices_2d['x1'], vertices_2d['x2'], vertices_2d['x3'], vertices_2d['x4']]
    y_list = [vertices_2d['y1'], vertices_2d['y2'], vertices_2d['y3'], vertices_2d['y4']]
    x_min = Decimal(min(x_list)).quantize(Decimal('0.00'))
    x_max = Decimal(max(x_list)).quantize(Decimal('0.00'))
    y_min = Decimal(min(y_list)).quantize(Decimal('0.00'))
    y_max = Decimal(max(y_list)).quantize(Decimal('0.00'))
    box_2d = [x_min, y_min, x_max, y_max]
    return box_2d

def cal_dim_loc(vertices_3d):
    '''
    Calculate the dimension of a vehicle from real 3D cuboid vertices.
    :param vertices_3d: {"x1": , "y1": , "z1": ,..., "x4": ,"y4": ,"z4":}
    :return: [h, w, l] in meters
    '''
    # since kpit has no real 3d points now, make fake points
    # front_left_top = np.array([vertices_3d['x1'], vertices_3d['y1'], vertices_3d['z1']])
    # front_right_top = np.array([vertices_3d['x2'], vertices_3d['y1'], vertices_3d['z2']])
    # front_left_bottom = np.array([vertices_3d['x1'], vertices_3d['y2'], vertices_3d['z1']])
    # back_left_top = np.array([vertices_3d['x3'], vertices_3d['y3'], vertices_3d['z3']])
    # back_right_bottom = np.array([vertices_3d['x4'], vertices_3d['y4'], vertices_3d['z4']])
    # width = np.linalg.norm(front_left_top - front_right_top)
    # height = np.linalg.norm(front_left_top - front_left_bottom)
    # length = np.linalg.norm(front_left_top - back_left_top)

    # also need to convert from np.array to float32
    # dimension_3d = [height, width, length]
    # location_3d = (front_left_top + back_right_bottom) / 2
    dimension_3d = [2.23, 3.35, 5.15]
    location_3d = [1.52, 3.55, 30.55]
    return dimension_3d, location_3d

def cal_rotation(vertices_3d):
    '''
    Calculate the orientation of a vehicle from real 3D cuboid vertices.
    :param vertices_3d: {"x1": , "y1": , "z1": ,..., "x4": ,"y4": ,"z4":}
    :return: alpha, rotation_y (in KITTI format)
    '''
    tv_alpha = -10
    tv_rotation = 0

    return tv_alpha, tv_rotation

def list_elem_2_str(list_ori):
    list_len = len(list_ori)
    list_str = []
    for i, elem in enumerate(list_ori):
        elem_str = str(elem)
        list_str.append(elem_str)
        if not i == (list_len - 1):
            list_str.append(' ')
    return list_str

def convert_label(label_per_vehicle):
    # [type, truncated, occluded, alpha, 2D_bbox(4), dimensions(3), location(3), rotation_y]
    label_1 = str(label_per_vehicle[0])  # type
    label_2 = str(Decimal(0).quantize(Decimal('0.00')))  # truncated
    label_3 = str(label_per_vehicle[1])  # occluded
    label_4 = str(label_per_vehicle[5])  # alpha
    label_5 = ''.join(list_elem_2_str(label_per_vehicle[2]))  # 2D_bbox
    label_6 = ''.join(list_elem_2_str(label_per_vehicle[3]))  # dimensions
    label_7 = ''.join(list_elem_2_str(label_per_vehicle[4]))  # location
    label_8 = str(label_per_vehicle[6])  # rotation_y
    label_kitti_format = [label_1, ' ', label_2, ' ', label_3, ' ', label_4, ' ',
                          label_5, ' ', label_6, ' ', label_7, ' ', label_8, ' ', '\n']
    return label_kitti_format

def convert_pair(pair):
    '''
    input:['/home/j0118570/Documents/KPIT/./annotation_results/Dataset-1A/Input-1/Scenario-1',
           './converted_labels/Dataset-1A/Input-1/labels/D1A_input_1_S1_000022.txt', '000022']
    output: training/image_2/Dataset-x/Input-x/xxxxx.png training/label_2/D1A_input_1_S1_000022.txt
    '''

    input_path = pair[0].split('/')[7] + '/' + pair[0].split('/')[8]
    img_name = pair[2]+'.png'
    # label_name = pair[1].split('/')[5]
    label_name = pair[2] + '.txt'

    img_path = 'training/image_2/' + input_path + '/' + img_name
    label_path = 'training/label_2/' + label_name
    converted_pair = img_path + ' ' + label_path + '\n'

    return converted_pair

def main():
    args = parse_arguments()
    work_dir = os.getcwd()
    video_base_path = args.input_path               # ./Dataset-x/Input-x/.
    save_path = args.save_path
    resized_image_path = os.path.join(work_dir, save_path, 'resized_images')
    img_label_pair = []
    input_images = []

    if os.path.isdir(video_base_path):
        scenario_folder_list = os.listdir(video_base_path)
        for scenario_folder_name in scenario_folder_list:
            scenario_folder_name = os.path.join(work_dir, video_base_path, scenario_folder_name)
            for i in os.listdir(scenario_folder_name):
                if os.path.splitext(i)[1] == '.json':
                    img_mark = os.path.splitext(i)[0]
                    label_file_path = os.path.join(scenario_folder_name, i)
                if i == 'Input_Frames':
                    image_path = os.path.join(scenario_folder_name, i)
                    input_images = os.listdir(image_path)

            # step1: resize images and save them
            if args.resize_image == "True":
                print('Resizing images now...')
                if not os.path.isdir(resized_image_path):
                    os.makedirs(resized_image_path)
                resize_image(input_images, image_path, resized_image_path)
            else:
                print('The process will not resize the images and save them!')


            # convert labels
            label_save_dir = os.path.join(save_path, 'labels')
            calib_save_dir = os.path.join(save_path, 'calib')
            if not os.path.isdir(label_save_dir):
                os.mkdir(label_save_dir)
            if not os.path.isdir(calib_save_dir):
                os.mkdir(calib_save_dir)

            with open(label_file_path, 'r') as label:
                label_per_scenario = json.load(label)
                print("{0} imgs in file {1}".format(len(label_per_scenario), label_file_path))
                print("Converting labels now ...")

                for label_per_img in tqdm(label_per_scenario):
                    img_name = label_per_img['image_name'].split('.')[0]
                    # Convert img name to be %6d
                    img_name = img_name.zfill(6)
                    img_labels = []

                    # convert labels
                    for idx in label_per_img['3d_info']:
                        # tv_id = idx['id']
                        tv_type = convert_type(idx['class'])
                        tv_occluded = convert_occl_label(idx['occlusion'])

                        # //todo: convert 2d & 3d coordinate to fit (1248, 384) image
                        tv_2d_box = cal_2d_box(idx['box_coords'])
                        # //todo: The followings are not annotated yet
                        tv_dimensions, tv_location = cal_dim_loc(idx['box_coords'])  # need to be 3d_box_coords
                        tv_alpha, tv_rotation = cal_rotation(idx['box_coords'])  # need to be 3d_box_coords

                        id_label = [tv_type, tv_occluded, tv_2d_box, tv_dimensions,
                                    tv_location, tv_alpha, tv_rotation]
                        img_labels.append(id_label)

                    # step2: save labels per image
                    label_save_path = os.path.join(label_save_dir, '{}.txt'.format(img_name))
                    with open(label_save_path, 'w') as outfile_img:
                        for vehicle in img_labels:
                            kitti_label = convert_label(vehicle)
                            outfile_img.write(''.join(kitti_label))

                    img_label_pair.append([scenario_folder_name, label_save_path, img_name])

                    # step3: save calibration files per image
                    calib_save_path = os.path.join(calib_save_dir, '{}.txt'.format(img_name))
                    with open(calib_save_path, 'w') as outfile_calib:
                        for line in open(os.path.join(save_path, '../../fov30_calib_sample.txt'), 'r'):
                            # pdb.set_trace()
                            outfile_calib.write(line)


        # step4: save image-label-path pairs in train.txt
        train_txt_save_path = os.path.join(save_path, 'train.txt')
        with open(train_txt_save_path, 'w') as outfile_train:
            for pair in img_label_pair:
                pair_line = convert_pair(pair)
                outfile_train.write(pair_line)
    else:
        raise IOError("Annatation results path not exsit!")


if __name__ == '__main__':
    main()
