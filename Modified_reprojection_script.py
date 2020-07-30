from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import json
# mpl.use('Agg')

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

# Please enter the below path as per data location 
image_input_path = '/home/j0118570/Documents/KPIT/annotation_results/sample_real_world_cuboid/Input-1/Scenario-1/Input_Frames/'  # path to image folder
output_path = 	   '' # path to output folder
json_file_path =   '/home/j0118570/Documents/KPIT/annotation_results/sample_real_world_cuboid/Input-1/Scenario-1/Sample_Task_Real_World_Cuboid.json' # path of respective JSON file

def json_parser(json_file_path):
    with open(json_file_path) as json_reader:
        json_data = json.load(json_reader)
        return json_data

json_data = json_parser(json_file_path)

intrinsic_mtx = np.asarray([[6274.71132, 0.0, 1920.0], [0.0, 6258.53001, 960.0], [0.0, 0.0, 1.0]]) # Camera matrix
extrinsic_mtx = np.asarray([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.3], [1.0, 0.0, 0.0, 0.0]]) # Rotation matrix | Translation vector

P2 = intrinsic_mtx.dot(extrinsic_mtx) # projection matrix calculated from intrinsic and extrinsic parameters

# Visualization Part
for obj_data in json_data:
    image = Image.open(image_input_path + obj_data['image_name'])
    fig = plt.figure(figsize=(38, 19))
    ax = fig.gca()
    ax.grid(False)
    ax.set_axis_off()
    ax.imshow(image)

    for data in obj_data['3d_info']:
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