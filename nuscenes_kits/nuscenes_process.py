#!/usr/bin/env python3

# Developed by Neng Wang
# Brief: nuscenes mutli-scan semantic labels generation
# The moving object extraction is inspired by MapMOS (https://github.com/PRBonn/MapMOS.git)
# This file is covered by the LICENSE file in the root of this project.


import os
import tqdm
import numpy as np
import multiprocessing as mp
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.splits import create_splits_logs
import importlib


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
CLASS_NAMES = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
def wirte_label(cla_label_name,attribute_label):
    label_final = []
    if cla_label_name == "vehicle.car":
        label_final.append("car")
        label_final.append(1) # class 
        label_final.append(0) # 0 for static and 1 for moving
    elif cla_label_name == "vehicle.truck":
        label_final.append("truck")
        label_final.append(2)
        label_final.append(0)
    elif cla_label_name == "vehicle.construction":
        label_final.append("construction_vehicle")
        label_final.append(3)
        label_final.append(0)
    elif cla_label_name =="vehicle.bus.bendy" or cla_label_name =="vehicle.bus.rigid":
        label_final.append("bus")
        label_final.append(4)
        label_final.append(0)
    elif cla_label_name =="vehicle.trailer":
        label_final.append("trailer")
        label_final.append(5)
        label_final.append(0)
    elif cla_label_name =="movable_object.barrier":
        label_final.append("barrier")
        label_final.append(6)
        label_final.append(0)
    elif cla_label_name =="vehicle.motorcycle":
        label_final.append("motorcycle")
        label_final.append(7)
        label_final.append(0)
    elif cla_label_name =="vehicle.bicycle":
        label_final.append("bicycle")
        label_final.append(8)
        label_final.append(0)
    elif cla_label_name =="human.pedestrian.adult" or cla_label_name =="human.pedestrian.child" or cla_label_name =="human.pedestrian.police_officer" or cla_label_name =="human.pedestrian.construction_worker":
        label_final.append("pedestrian")
        label_final.append(9)
        label_final.append(0)
    elif cla_label_name =="movable_object.trafficcone":
        label_final.append("traffic_cone")
        label_final.append(10)
        label_final.append(0)
    else:
        label_final.append("other")
        label_final.append(11)
        label_final.append(0)
    label_final.append(attribute_label)

    return label_final

def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw



def read_point_cloud(nusc, token: str):
    filename = nusc.get("sample_data", token)["filename"]
    load_point_cloud = importlib.import_module(
            "nuscenes.utils.data_classes"
        ).LidarPointCloud.from_file
    pcl = load_point_cloud(os.path.join(nusc.dataroot, filename))
    points = pcl.points.T[:, :4]
    return points.astype(np.float32)

def global_pose(nusc, token):
        sd_record_lid = nusc.get("sample_data", token)
        cs_record_lid = nusc.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ep_record_lid = nusc.get("ego_pose", sd_record_lid["ego_pose_token"])

        car_to_velo = transform_matrix(
            cs_record_lid["translation"],
            Quaternion(cs_record_lid["rotation"]),
        )
        pose_car = transform_matrix(
            ep_record_lid["translation"],
            Quaternion(ep_record_lid["rotation"]),
        )
        return pose_car @ car_to_velo

def trans_box_2_lidar_frame(nusc,box,lidar_token):
    #pose
    sd_record = nusc.get("sample_data", lidar_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
    box.translate(-np.array(pose_record["translation"]))
    box.rotate(Quaternion(pose_record["rotation"]).inverse)

    #  Move box to sensor coord system
    box.translate(-np.array(cs_record["translation"]))
    box.rotate(Quaternion(cs_record["rotation"]).inverse)
    return box

def create_kitti_label(nusc,train_scenes,split_log,train_type):

    # train_scenes = train_scenes[500:]
    save_dir = "./data/nuScenes_kitti"
    
    print(train_scenes)
    
    for scene_id in tqdm.tqdm(train_scenes):
        frame_idx = -1
        save_dir_train = os.path.join(save_dir,train_type,str(scene_id[-4:]))
        save_poses = os.path.join(save_dir_train,'poses.txt')
        path_velodyne = os.path.join(save_dir_train,'velodyne')
        path_label = os.path.join(save_dir_train,'labels')
        path_sem_label = os.path.join(save_dir_train,'sem_labels')
        path_bbox = os.path.join(save_dir_train,'boundingbox')
        os.makedirs(path_velodyne,exist_ok=True)
        os.makedirs(path_label,exist_ok=True)
        os.makedirs(path_bbox,exist_ok=True)
        os.makedirs(path_sem_label,exist_ok=True)
        print("========================")
        scene_tokens = [s["token"] for s in nusc.scene if s["name"] == scene_id][0]
        scene = nusc.get("scene", scene_tokens)
        log = nusc.get("log", scene["log_token"])
        moving_attributes = ["vehicle.moving", "cycle.with_rider", "pedestrian.moving"] # moving
        if True:
            start_sample_rec = nusc.get("sample", scene["first_sample_token"])
            sd_rec = nusc.get("sample_data", start_sample_rec["data"]["LIDAR_TOP"])
            cur_sd_rec = sd_rec
            sd_tokens = [cur_sd_rec["token"]]
            while cur_sd_rec["next"] != "":
                cur_sd_rec = nusc.get("sample_data", cur_sd_rec["next"])
                sd_tokens.append(cur_sd_rec["token"])
            
            key_frame_poses = []
            for idx in range(len(sd_tokens)):
                lidar_token = sd_tokens[idx]

                is_key_frame = nusc.get("sample_data", lidar_token)["is_key_frame"]
                sample_token = nusc.get("sample_data", lidar_token)["sample_token"]
                points = read_point_cloud(nusc,lidar_token)
                
                points_hom = np.hstack((points[:,:3], np.ones((len(points), 1))))

                lidar_pose = global_pose(nusc,lidar_token)[:3,:]

                if is_key_frame==True:
                    key_frame_poses.append(lidar_pose)
                global_points_hom = (global_pose(nusc,lidar_token) @ points_hom.T).T

                if is_key_frame==True:
                    frame_bbox = []
                    frame_idx = frame_idx+1
                    lidarseg_labels_filename = nusc.get('lidarseg', lidar_token)['filename']
                    label_path = os.path.join(nusc.dataroot,lidarseg_labels_filename)
                    annotated_data = np.fromfile(str(label_path), dtype=np.uint8, count=-1).reshape([-1]) # semantic label
                    path_label_frame = os.path.join(path_sem_label,str(frame_idx).zfill(6)+'.label')
                    #save single-scan semantic label
                    annotated_data.astype(np.uint8).tofile(path_label_frame)

                    annotation_tokens = nusc.get("sample", sample_token)["anns"]
                    for annotation_token in annotation_tokens:
                        attribute_bbox = np.zeros((7))
                        annotation = nusc.get("sample_annotation", annotation_token)
                        attribute_token = annotation["attribute_tokens"]
                        if len(attribute_token)>0 and nusc.get("attribute", attribute_token[0])["name"] in moving_attributes:
                            box = nusc.get_box(annotation_token)
                            box_center = box.center
                            box_rotation = box.rotation_matrix
                            box_wlh = box.wlh
                            name = box.name

                            box_pose = np.vstack(
                                [
                                    np.hstack([box_rotation, box_center.reshape(-1, 1)]),
                                    np.array([[0, 0, 0, 1]]),
                                ]
                            )
                            local_points = (np.linalg.inv(box_pose) @ global_points_hom.T).T[:, :3]
                            abs_local_points = np.abs(local_points)
                            mask = abs_local_points[:, 0] < box_wlh[1] / 2
                            mask = np.logical_and(mask, abs_local_points[:, 1] <= box_wlh[0] / 2)
                            mask = np.logical_and(mask, abs_local_points[:, 2] <= box_wlh[2] / 2)
                            if len(annotated_data[mask])<10:
                                continue
                            # print("name:",name,",,,,attribute:",nusc.get("attribute", attribute_token[0])["name"])
                            if nusc.get("attribute", attribute_token[0])["name"]=="vehicle.moving":
                                if name=="vehicle.car":
                                    annotated_data[mask] = 32 # moving car
                                elif name=="vehicle.bus.bendy" or name=="vehicle.bus.rigid":
                                    annotated_data[mask] = 33 # moving bus
                                elif name == "vehicle.truck":
                                    annotated_data[mask] = 34 # moving truck
                                elif name=="vehicle.construction":
                                    annotated_data[mask] = 35 # moving construction
                                elif name=="vehicle.trailer":
                                    annotated_data[mask] = 36 # moving trailer
                            elif nusc.get("attribute", attribute_token[0])["name"]=="cycle.with_rider":
                                if name=="vehicle.motorcycle":
                                    annotated_data[mask] = 37 # moving motorcyclist
                                elif name=="vehicle.bicycle":
                                    annotated_data[mask] = 38 # moving cyclist
                            elif nusc.get("attribute", attribute_token[0])["name"]=="pedestrian.moving":
                                annotated_data[mask] = 39     # moving pedestrian
                            
                            # moving bbox in vehicle frame
                        box_trans = trans_box_2_lidar_frame(nusc,nusc.get_box(annotation_token),lidar_token)
                        yaw = quaternion_yaw(box_trans.orientation)
                        # if yaw <-np.pi/2:
                        #     yaw = yaw + np.pi
                        # if yaw > np.pi/2 and yaw < np.pi*1.5 :
                        #     yaw = yaw - np.pi
                        # if yaw>1.5*np.pi and yaw< np.pi*2:
                        #     yaw = yaw - 2*np.pi
                        attribute_bbox[0:3] =  box_trans.center
                        attribute_bbox[3] =  box_trans.wlh[1]
                        attribute_bbox[4] =  box_trans.wlh[0]
                        attribute_bbox[5] =  box_trans.wlh[2]
                        attribute_bbox[6] = yaw
                        frame_bbox.append(wirte_label(box_trans.name,attribute_bbox))
                        
                    path_velodyne_frame = os.path.join(path_velodyne,str(frame_idx).zfill(6)+'.bin')
                    path_label_frame = os.path.join(path_label,str(frame_idx).zfill(6)+'.label')
                    path_bbox_frame = os.path.join(path_bbox,str(frame_idx).zfill(6)+'.npy')

                    #save velodyne
                    points.astype(np.float32).tofile(path_velodyne_frame)
                    #save multi-scan semantic label
                    annotated_data.astype(np.uint8).tofile(path_label_frame)
                    #save bbox
                    boundingbox_label = np.array(frame_bbox,dtype=object)
                    np.save(path_bbox_frame,boundingbox_label) 
            # save keyframe poses
            with open(save_poses, 'w') as file:
                for pose in key_frame_poses:
                    pose_str = ' '.join(map(str, pose.flatten().tolist()))
                    file.write(pose_str + '\n')

            print("sd_tokens:",len(sd_tokens))
        else:
            print("!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!")
        
        # print("scene:",scene,",log:",log)

def main(data_path):
    split: str = "train"
    lidar_name: str = "LIDAR_TOP"
    version = "v1.0-trainval"

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    split_logs = create_splits_logs(split, nusc)
    print(split_logs)

    create_kitti_label(nusc,train_scenes,split_logs,'train')
    create_kitti_label(nusc,val_scenes,split_logs,'val')

if __name__=="__main__":
    data_path = "/home/wangneng/DataFast/nuscences"  # directory of nuscenes
    main(data_path)