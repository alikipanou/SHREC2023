import os.path as osp
import random
import glob
import numpy as np
import torch.utils.data
import laspy

import pandas as pd
import re
import open3d as o3d

#####     library versions  ######

# torch==1.13.1
# laspy==2.3.0
# open3d==0.11.2
# pandas==1.3.2
# numpy== 1.19.5


# Synthetic Dataset consists of 639 object pairs

class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        #subset = 'train',
        point_limit=2048,
        shape = 'cylinder',
        clearance = 1
    ):
        super(SyntheticDataset, self).__init__()

        self.point_limit = point_limit
        self.shape = shape
        self.clearance = clearance

        #one hot encode labels (some labels are misspelled) 
        self.class_dict = {'added' : np.array([1,0,0,0,0]), 'removed': np.array([0,1,0,0,0]), 'nochange' : np.array([0,0,1,0,0]),
                        'change' : np.array([0,0,0,1,0]), 'color_change': np.array([0,0,0,0,1]), 'adeed': np.array([1,0,0,0,0]),
                           'changee': np.array([0,0,0,1,0]), 'nohcange': np.array([0,0,1,0,0]), 'remoced' : np.array([0,1,0,0,0]),
                           'reomved': np.array([0,1,0,0,0])}

        # 'train' or 'val'
        #self.subset = subset

        # get all paths from time_a, time_b and classifications in lists
        self.files_time_a = glob.glob(osp.join('time_a', '*.las'))
        self.files_time_a.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        self.files_time_b = glob.glob(osp.join('time_b', '*.las'))
        self.files_time_b.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
 
        self.classification_files = glob.glob(osp.join('labeled_point_lists_train_syn', '*.csv'))
        self.classification_files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


        self.lengths = []
        self.total_points_of_interest = 0
        previous = 0

        self.labels_count = {'added' : 0, 'removed': 0, 'nochange' : 0,
                        'change' : 0, 'color_change': 0, 'adeed': 0,
                           'changee': 0, 'nohcange': 0, 'remoced' : 0,
                           'reomved': 0}
        # list with starting and ending index of points of interest that belong in a single csv file
        for i, path in enumerate(self.classification_files):
            df = pd.read_csv(path)
            self.lengths.append(len(df) + previous)
            previous = self.lengths[i]
            self.total_points_of_interest += len(df)

            for j in range(len(df)):
                df_row = df.iloc[j]
                self.labels_count[df_row['classification']] += 1

        # dataset is highly unbalanced
        # create weights for each class
        self.labels_count['added'] += self.labels_count['adeed']
        self.labels_count['removed'] += (self.labels_count['remoced'] + self.labels_count['reomved'])
        self.labels_count['change'] += self.labels_count['changee']
        self.labels_count['nochange'] += self.labels_count['nohcange']

        del self.labels_count['adeed']
        del self.labels_count['remoced']
        del self.labels_count['reomved']
        del self.labels_count['changee']
        del self.labels_count['nohcange']
        
        num_classes = 5 
        self.weights = [self.total_points_of_interest / (num_classes * label_count) for label_count in self.labels_count.values()]

       
        self.weights = np.asarray(self.weights).astype(np.float32)
        #print(self.weights)
 

    def _load_point_cloud_from_las_file(self, path):
        input_las = laspy.read(path)
    
        las_scaleX = input_las.header.scale[0]
        las_offsetX = input_las.header.offset[0]
        las_scaleY = input_las.header.scale[1]
        las_offsetY = input_las.header.offset[1]
        las_scaleZ = input_las.header.scale[2]
        las_offsetZ = input_las.header.offset[2]

        # calculating coordinates
        p_X = np.array((input_las.X * las_scaleX) + las_offsetX)
        p_Y = np.array((input_las.Y * las_scaleY) + las_offsetY)
        p_Z = np.array((input_las.Z * las_scaleZ) + las_offsetZ)

        # colors and normalization in range (0,1)
        red = np.array(input_las.red)
        red  = (red -  np.min(red)) / (np.max(red))
        
        green = np.array(input_las.green)
        green  = (green - np.min(green)) / (np.max(green) - np.min(green))
        
        blue = np.array(input_las.blue)
        blue  = (blue - np.min(blue)) / (np.max(blue) - np.min(blue))

        points = np.vstack((p_Z, p_X, p_Y ,red, green, blue)).T
        
        return points
    
    def _extract_area(self, full_cloud,center,clearance = 2.5,shape= 'cylinder'):
        if shape == 'square':
            x_mask = ((center[0]+clearance)>full_cloud[:,0]) &   (full_cloud[:,0] >(center[0]-clearance))
            y_mask = ((center[1]+clearance)>full_cloud[:,1]) &   (full_cloud[:,1] >(center[1]-clearance))
            mask = x_mask & y_mask
        elif shape == 'cylinder':
            mask = np.linalg.norm(full_cloud[:,:2]-center,axis=1) <  clearance

        area = full_cloud[mask]
        
        # fix axes
        points_z = area[:,0]
        points_x = area[:,1]
        points_y = area[:,2]

        points = np.vstack((points_x, points_y, points_z, area[:,3], area[:,4], area[:,5])).T
        return points

    def _random_subsample(self, points, point_limit = 2048):
        if points.shape[0]==0:
            print('No points found at this center replacing with dummy')
            points = np.zeros((1,points.shape[1]))
        #No point sampling if already 
        if self.point_limit < points.shape[0]:
            random_indices = np.random.choice(points.shape[0],self.point_limit, replace=False)
            points = points[random_indices,:]
        return points
    
    def _make_open3d_point_cloud(self, points, colors=None, normals=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd

    def _visualize_object_pair(self, index):
        
        data_dict = self.__getitem__(index)

        pcd1 = self._make_open3d_point_cloud(data_dict['ref_points'], data_dict['ref_feats'])

        #translate 2nd object
        data_dict['src_points'][:,0] += 10
        pcd2 = self._make_open3d_point_cloud(data_dict['src_points'], data_dict['src_feats'])

        #visualize
        print("Scene id = " + data_dict['scene_id'])
        print("Class = " + data_dict['class'])
        
        geometries = []
        geometries.append(pcd1)
        geometries.append(pcd2)
        o3d.visualization.draw_geometries(geometries)

    def __getitem__(self, index):
        data_dict = {}

        file_index = np.searchsorted(self.lengths, index, side='right')

        #full clouds
        points_a = self._load_point_cloud_from_las_file(self.files_time_a[file_index])
        points_b = self._load_point_cloud_from_las_file(self.files_time_b[file_index])
        classification_df = pd.read_csv(self.classification_files[file_index])

        #extract object pair from full clouds
        if (file_index == 0):
            inner_index = index
        else:
            inner_index = index - self.lengths[file_index - 1]
            
        df_row = classification_df.iloc[inner_index]
        center = np.array([df_row['z'],df_row['x']])
        object_a = self._random_subsample(self._extract_area(points_a, center, clearance = self.clearance, shape = self.shape), point_limit = self.point_limit)
        object_b = self._random_subsample(self._extract_area(points_b, center, clearance = self.clearance, shape = self.shape), point_limit = self.point_limit)
        classification  = df_row['classification']

        
        object_a_points = object_a[:,:3]
        object_b_points = object_b[:,:3]

        object_a_rgb = object_a[:,3:]
        object_b_rgb = object_b[:,3:]

        data_dict['scene_id'] = re.split(r'(\d+)', self.files_time_a[file_index])[1]
        data_dict['center'] = np.array([df_row['x'],df_row['y'], df_row['z']]).astype(np.float32)
        data_dict['ref_points'] = object_a_points.astype(np.float32)
        data_dict['src_points'] = object_b_points.astype(np.float32)
        data_dict['ref_feats'] = object_a_rgb.astype(np.float32)
        data_dict['src_feats'] = object_b_rgb.astype(np.float32)
        data_dict['class'] = classification
        data_dict['label'] = self.class_dict[classification].astype(np.float32)
        

        return data_dict

    def __len__(self):
        return self.total_points_of_interest


dataset = SyntheticDataset(shape = 'square')
#print(len(dataset))
dataset._visualize_object_pair(329)

