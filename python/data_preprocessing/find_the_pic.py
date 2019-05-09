# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:23:54 2019
"""

import SimpleITK as sitk
import numpy as np
import csv
import os
import glob
import pandas as pd
import scipy.io
import settings

# FIX CRASH #
import matplotlib
matplotlib.use("TkAgg")
# --------- #
import matplotlib.pyplot as plt

# 尝试调用 tqdm 如果失败就用一个lambda表达式代替
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

# 定义工作目录和文件位置
working_dir = os.path.split(os.path.realpath(__file__))[0]
annotations_path = os.path.join(working_dir, "annotations.csv")
output_path = os.path.join(working_dir, "output")

luna_root_path = settings.luna_root_path
file_list = []
for subpath in os.listdir(luna_root_path):
	file_list.extend(glob.glob(os.path.join(luna_root_path, subpath, "*.mhd")))

# 遗留函数，作用未知
def matrix2int16(matrix):
    '''
    matrix must be a numpy array NXN
    Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

# 遗留函数，产生一个512x512的遮罩矩阵，其中要显示的区域值为1，其他为0
def make_mask(center,diam,z,width,height,spacing,origin):
    '''
    Center : centers of circles px -- list of coordinates x,y,z
    diam : diameters of circles px -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width],dtype=np.int8) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls

    SIZE = 50 # 数据切片大小 50*50
    if diam/spacing[0] > SIZE:
        print("this nodule is bigger than mask size!")
        return mask

    v_center = (center-origin)/spacing
    # v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0])-SIZE//2 + 1]) # 加一保证 size 为 50x50
    v_xmax = np.min([width-1,int(v_center[0])+SIZE//2])
    v_ymin = np.max([0,int(v_center[1])-SIZE//2 + 1]) # 加一保证 size 为 50x50
    v_ymax = np.min([height-1,int(v_center[1])+SIZE//2])
    # ! 注意这里 nodule 的中心在中心偏左上角一个像素的位置，因为图像是 50x50 没有正中心的像素点

    # Convert back to world coordinates for distance calculation
        #x_data = [x*spacing[0]+origin[0] for x in range(width)]
        #y_data = [x*spacing[1]+origin[1] for x in range(height)]

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    for v_x in v_xrange:                #修改两个范围来规范遮罩的大小
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1
    '''
    #圆形遮罩
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam: #np.linalg.norm求范数，此处即求结节中心到遮罩中心坐标的距离，如果小于等于结节直径，则填黑
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    '''

    return mask

# 返回一个有三个元素的tuple，前两个元素是要切割区域的左上角点的矩阵索引位置(y,x)
# 第三个元素是一个标志位，用于警告上级函数nodule是否不在图片中心（因为太靠边缘导致）
def get_mask_position(center,diam,z,width,height,spacing,origin):
    SIZE = 50 # 数据切片大小 50*50
    ORIGIN_SIZE = 512
    notInTheCenterFlag = 0
    if diam/spacing[0] > SIZE:
        print("this nodule is bigger than 50x50 size!")

    v_center = (center-origin)/spacing
    v_xmin = int(v_center[0])-SIZE//2 + 1 # 加一保证 size 为 50x50
    v_ymin = int(v_center[1])-SIZE//2 + 1 # 加一保证 size 为 50x50

    lb = 0                  # low boundary
    ub = ORIGIN_SIZE-SIZE-1 # up boundary
    if v_xmin < lb | v_ymin < lb | v_xmin > ub | v_ymin > ub:
        # nodule在边缘！nodule将不会在图片中心,返回 flag 通知上级函数
        notInTheCenterFlag = 1
    v_xmin = lb if v_xmin < lb else v_xmin
    v_ymin = lb if v_ymin < lb else v_ymin
    v_xmin = ub if v_xmin > ub else v_xmin
    v_ymin = ub if v_ymin > ub else v_ymin

    return (v_ymin,v_xmin,notInTheCenterFlag)

# with each file
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

# Get locations of the nodes
df_node = pd.read_csv(annotations_path)
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_name))
df_node = df_node.dropna()


# Looping over the image files
sml_img_data = []
v_axis_annotation_data = [] # 用于储存新的坐标系下的标记数据
for fcount, img_file in enumerate(tqdm(file_list)):
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            seriesuid = cur_row["seriesuid"]
            # keep 3 slices
            imgs = np.ndarray([3,height,width],dtype=np.int16)
            sml_imgs = np.ndarray([3,50,50],dtype=np.int16)
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)


            annotation = [seriesuid]
            annotation.extend(v_center.tolist())
            annotation.append(diam)
            v_axis_annotation_data.append(annotation)

            for i, i_z in enumerate(np.arange(int(v_center[2])-1, int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                mask = get_mask_position(center, diam, i_z*spacing[2]+origin[2], width, height, spacing, origin)
                temp_img = img_array[i_z]
                small_img = temp_img[mask[0]:mask[0]+50,mask[1]:mask[1]+50] # nodule 50x50图片
                imgs[i] = temp_img
                sml_imgs[i] = small_img
                if mask[2] == 1:
                    print("nodule 不在图片的中心！")
            sml_img_data.append(sml_imgs)

# 将切出的数据存储到 .npy 和 .mat中，方便后续使用
sml_img_data = np.asarray(sml_img_data)
np.save(os.path.join(output_path, 'sml_img_data'), sml_img_data)
scipy.io.savemat(os.path.join(output_path, 'sml_img_data.mat'), {'sml_img_data': sml_img_data})

# 将转换坐标系后的标记存储到 annotation_v.csv
with open(os.path.join(working_dir,'output','annotation_v.csv'), 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["seriesuid","coordX","coordY","coordZ","diameter_mm"])
    writer.writerows(v_axis_annotation_data)
csvFile.close()