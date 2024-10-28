#!/usr/bin/env python3

# Developed by Neng Wang
# Brief: generating instance bounding box from semantic point cloud
# This file is covered by the LICENSE file in the root of this project.



import open3d as o3d
import numpy as np
import numpy.linalg as LA
import os
from tqdm import *
from colorsys import hls_to_rgb
import copy
import hdbscan  # optional
import pcl
import argparse
from LShapeUtils import LShapeFitting


label_map = {
  0 : 0,     # "unlabeled"      
  1 : 0,     # "outlier"        
  10: 1,     # "car"            
  11: 2,     # "bicycle"        
  13: 3,     # "bus"            
  15: 4,     # "motorcycle"      
  16: 5,     # "on-rails"        
  18: 6,     # "truck"           
  20: 7,     # "other-vehicle"   
  30: 8,     # "person"          
  31: 9,     # "bicyclist"       
  32: 10,     # "motorcyclist"    
  40: 0,     # "road"            background
  44: 0,    # "parking"          background
  48: 0,    # "sidewalk"         background
  49: 0,    # "other-ground"     background
  50: 0,    # "building"         background
  51: 0,    # "fence"            background
  52: 0,     # "other-structure" background
  60: 0,     # "lane-marking"    background
  70: 0,    # "vegetation"       background
  71: 0,    # "trunk"            background
  72: 0,    # "terrain"          background
  80: 0,    # "pole"             background
  81: 0,    # "traffic-sign"     background
  99: 0,     # "other-object"    background
  252: 11,    # "moving-car"          
  253: 19,    # "moving-bicyclist"    
  254: 18,    # "moving-person"       
  255: 20,    # "moving-motorcyclist" 
  256: 15,    # "moving-on-rails"     
  257: 13,    # "moving-bus"          
  258: 16,    # "moving-truck"        
  259: 12    # "moving-other"        
}


def load_scan(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32).reshape((-1,4))
    return scan

def load_labels(label_path):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    assert ((sem_label + (inst_label << 16) == label).all())
    return sem_label, inst_label

def gen_color_map(n):
  """ generate color map given number of instances
  """
  colors = []
  for i in np.arange(0., 360., 360. / n):
    h = i / 360.
    l = (50 + np.random.rand() * 10) / 100.
    s = (90 + np.random.rand() * 10) / 100.
    colors.append(hls_to_rgb(h, l, s))

  return np.array(colors)

def clusters_hdbscan(points_set):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )
    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True) 
    cluster_info = np.array(list(zip(lbls[1:], counts[1:]))) 
    cluster_info = cluster_info[cluster_info[:,1].argsort()] 

    clusters_labels = cluster_info[::-1][:, 0]  
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1 

    return labels


def wirte_label(cla_label,attribute_label):
    label_final = []
    if cla_label ==1:
        label_final.append("car")
        label_final.append(1) # category
        label_final.append(0) # motion attribute
    elif cla_label == 2:
        label_final.append("bicycle")
        label_final.append(2)
        label_final.append(0)
    elif cla_label == 3:
        label_final.append("bus")
        label_final.append(3)
        label_final.append(0)
    elif cla_label ==4:
        label_final.append("motorcycle")
        label_final.append(4)
        label_final.append(0)
    elif cla_label ==5:
        label_final.append("onrails")
        label_final.append(5)
        label_final.append(0)
    elif cla_label ==6:
        label_final.append("truck")
        label_final.append(6)
        label_final.append(0)
    elif cla_label ==7:
        label_final.append("othervehicle")
        label_final.append(7)
        label_final.append(0)
    elif cla_label ==8:
        label_final.append("person")
        label_final.append(8)
        label_final.append(0)
    elif cla_label ==9:
        label_final.append("bicyclist")
        label_final.append(9)
        label_final.append(0)
    elif cla_label ==10:
        label_final.append("motorcyclist")
        label_final.append(10)
        label_final.append(0)
    elif cla_label ==11:
        label_final.append("car")
        label_final.append(1) 
        label_final.append(1)
    elif cla_label ==12:
        label_final.append("other")
        label_final.append(12)  
        label_final.append(1)
    elif cla_label ==13:
        label_final.append("bus")
        label_final.append(3)
        label_final.append(1)
    elif cla_label ==15:
        label_final.append("onrails")
        label_final.append(5)
        label_final.append(1)
    elif cla_label ==16:
        label_final.append("truck")
        label_final.append(6)
        label_final.append(1)
    elif cla_label ==18:
        label_final.append("person")
        label_final.append(8)
        label_final.append(1)
    elif cla_label ==19:
        label_final.append("bicyclist")
        label_final.append(9)
        label_final.append(1)
    elif cla_label ==20:
        label_final.append("motorcyclist")
        label_final.append(10)
        label_final.append(1)
    label_final.append(attribute_label)

    return label_final
    

def PointsCluster(mapped_labels,scan):
    pointscluster = []
    inst_id = 1
    for sem_label in range(1,21):
        selected_labels = np.zeros(mapped_labels.shape) 
        if sem_label in mapped_labels: # points with same semantic label
            selected_labels[mapped_labels==sem_label]=1
            selected_points = scan[selected_labels==1,:3]

            if sem_label in [1,3,5,6,7,11,13,15,16]:
                cluster_tolerance = 0.5
                min_size = 60
            else:
                cluster_tolerance = 0.8
                min_size = 20
            cloud = pcl.PointCloud(selected_points)
            tree = cloud.make_kdtree()
            segment = cloud.make_EuclideanClusterExtraction()
            segment.set_ClusterTolerance(cluster_tolerance)
            segment.set_MinClusterSize(min_size)
            segment.set_MaxClusterSize(50000)
            segment.set_SearchMethod(tree)
            cluster_indices = segment.Extract()
            for j, indices in enumerate(cluster_indices):
                inst_cluster = np.zeros((len(indices), 3),dtype=np.float32)
                inst_cluster = selected_points[np.array(indices), 0:3]
                inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), sem_label, dtype=np.uint32)), axis=1)
                inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                inst_id =inst_id+1
                pointscluster.append(inst_cluster)


    return pointscluster
    

def bev_boundingbox(data_3d):
    data = np.zeros(data_3d.shape)
    data[:,:2] = data_3d[:,:2]
    data =data.transpose()

    means = np.mean(data, axis=1)
    cov = np.cov(data)
    eval, evec = LA.eig(cov)
    centered_data = data - means[:,np.newaxis]
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(centered_data[0, :]), np.max(centered_data[0, :]), np.min(centered_data[1, :]), np.max(centered_data[1, :]), np.min(centered_data[2, :]), np.max(centered_data[2, :])

    aligned_coords = np.matmul(evec.T, centered_data)
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])


    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                        [y1, y2, y2, y1, y1, y2, y2, y1],
                                                        [z1, z1, z1, z1, z2, z2, z2, z2]])

    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]

    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis] 
    center = np.zeros(3)
    center[0] = np.sum(rrc[0,:])/8
    center[1] = np.sum(rrc[1,:])/8
    center[2] = np.sum(rrc[2,:])/8
    extent = np.zeros(3)
    extent[0] = abs(xmax-xmin)
    extent[1] = abs(ymax-ymin)
    extent[2] = abs(zmax-zmin)

    theta_cos = np.arccos(evec[0,0])
    theta_sin = np.arcsin(evec[1,0])
    if theta_sin <= 0:
        theta = -theta_cos
    else:
        theta = theta_cos
    
    return  evec,center,extent,theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--view', action='store_true', help='true for visualization')
    parser.add_argument('--save', action='store_true', help='true for saving the bounding box label')
    parser.add_argument('--lshape', action='store_true', help='using lshape to refine the bounding box, specify for car')
    args = parser.parse_args()

    data_dir = args.data_path
    scan_folder = os.path.join(data_dir, 'velodyne')
    label_folder = os.path.join(data_dir,'labels')
    scan_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_folder)) for f in fn]
    scan_paths.sort()
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
    label_paths.sort()
    

    save_folder = os.path.join(data_dir, 'boundingbox')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for frame_idx in tqdm(range(0,len(scan_paths))):
        scan = load_scan(scan_paths[frame_idx])
        sem_label,_ = load_labels(label_paths[frame_idx])
        filename=label_paths[frame_idx][-12:-6] + '.npy'
        file_path = os.path.join(save_folder,filename)

        save_label_sigal_frame = []

        # label mapping
        mapped_labels = copy.deepcopy(sem_label)
        for k, v in label_map.items():
            mapped_labels[sem_label == k] = v

   
        background_selected = scan[mapped_labels==0,:3]
        back_ground_point = o3d.geometry.PointCloud()
        back_ground_point.points = o3d.utility.Vector3dVector(background_selected[:, :3])
        background_color = np.zeros((len(background_selected),3),dtype=np.float)
        background_color = background_color + 0.5
        back_ground_point.colors =o3d.utility.Vector3dVector(background_color)   

        # instance clustering
        pointcloud_cluster = PointsCluster(mapped_labels,scan)

        if args.view:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(back_ground_point)
            
            if len(pointcloud_cluster)!=0 :
                pointcloud_inst  = np.concatenate(pointcloud_cluster,axis=0)
                instance_varity = len(pointcloud_cluster)+1
                inst_label = pointcloud_inst[:,-1]
                pcd = o3d.geometry.PointCloud()
                color_map = gen_color_map(instance_varity)
                colors = color_map[inst_label.astype(int)]
                pcd.points = o3d.utility.Vector3dVector(pointcloud_inst[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(colors)

                vis.add_geometry(pcd)

        for j in range(0,len(pointcloud_cluster)):
            instance_attribute = np.zeros((7))
            pcd_instance = o3d.geometry.PointCloud()
            pcd_instance.points = o3d.utility.Vector3dVector(pointcloud_cluster[j][:, :3])
            aabb = pcd_instance.get_axis_aligned_bounding_box()
            aabb_center = aabb.get_center()
            aabb_extent = aabb.get_extent()
            bev_r,bev_center,bev_extent,theta = bev_boundingbox(pointcloud_cluster[j][:, :3])
            
            bounding_box_center = np.zeros(3)
            bouding_box_extent = np.ones(3)*1000 # init with a large value
            bounding_box_center[0] = bev_center[0]
            bounding_box_center[1] = bev_center[1]
            bounding_box_center[2] = aabb_center[2]
            bouding_box_extent[0] = bev_extent[0]
            bouding_box_extent[1] = bev_extent[1]
            bouding_box_extent[2] = aabb_extent[2]

            # LShapeFitting can refine the box of car, but it is time consumping, occasionally producing incorrect results for bicyclist
            lshape_box_center = np.zeros(3)
            lshape_box_extent = np.ones(3)*1000 # init with a large value
            if args.lshape:
                lshapefitting = LShapeFitting()
                rect_bbox = lshapefitting.fitting(pointcloud_cluster[j][:, :3])
                l_shapebox = rect_bbox.calc_rect_contour()
                
                lshape_box_center[0] = l_shapebox[0]
                lshape_box_center[1] = l_shapebox[1]
                lshape_box_center[2] = aabb_center[2]
                lshape_box_extent[0] = l_shapebox[2]
                lshape_box_extent[1] = l_shapebox[3]
                lshape_box_extent[2] = aabb_extent[2]
                lshaep_theta = l_shapebox[4]

            V_aabb = aabb_extent[0]*aabb_extent[1]*aabb_extent[2]
            V_obb = bouding_box_extent[0]*bouding_box_extent[1]*bouding_box_extent[2]
            V_lshape = lshape_box_extent[0]*lshape_box_extent[1]*lshape_box_extent[2]

            print("V_aabb:",V_aabb,"V_obb:",V_obb,"V_lshape:",V_lshape)


            if min(V_aabb,V_obb,V_lshape) == V_aabb:
                instance_attribute[0:3] = aabb_center
                instance_attribute[3:6] = aabb_extent 
                instance_attribute[6] = 0       
                if args.view:
                    aabb.color = [0.0,0,0]
                    vis.add_geometry(aabb)
            elif min(V_aabb,V_obb,V_lshape) == V_obb:
                Rotation_eye = np.eye(3)
                obb =  o3d.geometry.OrientedBoundingBox(center=bounding_box_center,R=Rotation_eye,extent=bouding_box_extent)
                Rotation = obb.get_rotation_matrix_from_xyz((0,0,theta))

                obb_rotation =  o3d.geometry.OrientedBoundingBox(center=bounding_box_center,R=Rotation,extent=bouding_box_extent)
                obb_rotation.color = [0,0,0]
                instance_attribute[0:3] = bounding_box_center
                instance_attribute[3:6] = bouding_box_extent 
                instance_attribute[6] = theta
                if args.view:
                    vis.add_geometry(obb_rotation)
            elif min(V_aabb,V_obb,V_lshape) == V_lshape:
                Rotation_eye = np.eye(3)
                obb =  o3d.geometry.OrientedBoundingBox(center=lshape_box_center,R=Rotation_eye,extent=lshape_box_extent)
                Rotation = obb.get_rotation_matrix_from_xyz((0,0,lshaep_theta))

                obb_rotation =  o3d.geometry.OrientedBoundingBox(center=lshape_box_center,R=Rotation,extent=lshape_box_extent)
                obb_rotation.color = [0,0,0]
                instance_attribute[0:3] = lshape_box_center
                instance_attribute[3:6] = lshape_box_extent 
                instance_attribute[6] = lshaep_theta 
                if args.view:
                    vis.add_geometry(obb_rotation)

            if instance_attribute[6] <-np.pi/2: # angle range [-pi/2,pi/2]
                instance_attribute[6] = instance_attribute[6] + np.pi
            if instance_attribute[6] > np.pi/2 and instance_attribute[6] < np.pi*1.5 :
                instance_attribute[6] = instance_attribute[6] - np.pi
            if instance_attribute[6]>1.5*np.pi and instance_attribute[6] < np.pi*2:
                instance_attribute[6] = instance_attribute[6] - 2*np.pi


            semeantic_label = pointcloud_cluster[j][0,-2]
            save_label_sigal_frame.append(wirte_label(semeantic_label,instance_attribute))

        if args.save:
            boundingbox_label = np.array(save_label_sigal_frame,dtype=object)
            np.save(file_path,boundingbox_label) 


        if args.view:
            print("Showing visualization frame:",frame_idx)
            while True:
                vis.update_geometry(pcd)
                vis.update_geometry(back_ground_point)
                vis.poll_events()
                vis.update_renderer()
    

    