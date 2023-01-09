import argparse
import os 
import cv2
import random
from tqdm import tqdm
from xml.etree import ElementTree as ET

def crop_Kitti2ROI(labelPath, augment=False) -> bool: 

    assert os.path.isdir(labelPath), "%s does not exist."%labelPath
    
    anno_folder = labelPath.replace(labelPath.split('/')[-1], 'roi_annotations')
    if not os.path.exists(anno_folder):
        os.mkdir(anno_folder)
    assert len(os.listdir(anno_folder))<1, "Annotations already exist."
    
    img_folder = labelPath.replace(labelPath.split('/')[-1], 'images')
    roi_img_folder = labelPath.replace(labelPath.split('/')[-1], 'roi_images')
    if not os.path.exists(roi_img_folder):
        os.mkdir(roi_img_folder)
    assert len(os.listdir(roi_img_folder))<1, "ROI xmls already exist."
    
    txt_folder = labelPath.replace(labelPath.split('/')[-1], 'roi_labels')
    if not os.path.exists(txt_folder):
        os.mkdir(txt_folder)
    assert len(os.listdir(txt_folder))<1, "ROI labels already exist."
    
    file_lt = os.listdir(labelPath)
    file_lt = [f for f in file_lt if file_lt.index(f)%FREQUENCY==0] # choose every 5 frames
    cnt = 0
    pbar = tqdm(file_lt, desc='Files:')
    for file in pbar:
        if cnt>=TARGET_NUM:
            break 
        file_id = file.split('.')[0]
        file_path = os.path.join(labelPath, file)
        img_path = os.path.join(img_folder, file.replace('.txt', '.png'))
        img_mat = cv2.imread(img_path)
        img_size = img_mat.shape
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                if cnt>=TARGET_NUM:
                    break
                line_lt = line.split(' ')
                if not TRUNCATED:
                    if float(line_lt[1])!=0: # float from 0(non-truncated) to 1(truncated)
                        continue
                if not OCCLUDED:
                    if int(line_lt[2]): # 0=fully visible, 1=partly occluded, 2=largely occluded, 3=unknown
                        continue
                c = line_lt[0]
                if c not in CLASS_NAMES:
                    continue
                y, x = float(line_lt[13]), float(line_lt[11])
                if y<Y_RANGE[0] or y>Y_RANGE[1] or abs(x)>X_RANGE:
                    continue
                else:
                    y, x = y/Y_RANGE[1], (x+X_RANGE)/(2*X_RANGE) # normalize x,y into 0~1
                bbox = [float(bi) for bi in line_lt[4:8]]
                bw, bh = bbox[2]-bbox[0], bbox[3]-bbox[1]
                expansion_offsets = [-bw*0.25, -bh*0.25, bw*0.25, bh*0.25]
                roi_info = [int(bi+oi) for bi,oi in zip(bbox, expansion_offsets)]
                if roi_info[0] < 0:
                    roi_info[0] = 0
                if roi_info[1] < 0:
                    roi_info[1] = 0
                if roi_info[2]>img_size[1]-1:
                    roi_info[2] = img_size[1]-1
                if roi_info[3]>img_size[0]-1:
                    roi_info[3] = img_size[0]-1
                roi_size = [roi_info[2]-roi_info[0]+1, roi_info[3]-roi_info[1]+1, img_mat.shape[2]]
                bbox = [round(bbox[0]-roi_info[0]), round(bbox[1]-roi_info[1]),
                        round(bbox[2]-roi_info[0]), round(bbox[3]-roi_info[1])]
                if bbox[0]<0 or bbox[1]<0 or bbox[2]>roi_size[0] or bbox[3]>roi_size[1]:
                    continue
                roi_img_path = os.path.join(roi_img_folder, file_id+'_%s.jpg'%str(i))
                anno_path = os.path.join(anno_folder, file_id+'_%s.xml'%str(i))
                txt_path = os.path.join(txt_folder, file_id+'_%s.txt'%str(i))
                roi_mat = img_mat[roi_info[1]:roi_info[3]+1, roi_info[0]:roi_info[2]+1, :]
                cv2.imwrite(roi_img_path, roi_mat)
                write_XML(anno_path, roi_img_path, roi_size, c, roi_info, bbox, y, x)
                write_txt(txt_path, img_size, roi_size, c, bbox, y, x, roi_info)
                cnt+=1
                if augment:
                    move_mat = [[0,1,0,1],
                                [0,-1,0,-1],
                                [-1,0,-1,0],
                                [1,0,1,0]] # up,down,left,right 
                    for j in range(len(move_mat)):
                        if cnt>=TARGET_NUM:
                            break 
                        seed = random.uniform(0.4, 0.9)
                        xyxy = [bbox[0], bbox[1], bbox[0], bbox[1]]
                        m = [int(mi*seed*xyi) for mi,xyi in zip(move_mat[j], xyxy)]
                        roi_info_aug = [ri+mi for ri,mi in zip(roi_info, m)]
                        if roi_info_aug[0]<0:
                            roi_info_aug[0] = 0
                        if roi_info_aug[1]<0:
                            roi_info_aug[1] = 0
                        if roi_info_aug[2]>img_size[1]-1:
                            roi_info_aug[2] = img_size[1]-1
                        if roi_info_aug[3]>img_size[0]-1:
                            roi_info_aug[3] = img_size[0]-1

                        bbox_aug = [bi-mi for bi,mi in zip(bbox, m)]
                        if bbox_aug[0]<0 or bbox_aug[1]<0 or bbox_aug[2]>roi_size[0] or bbox_aug[3]>roi_size[1]:
                            continue
                        roi_size_aug = [roi_info_aug[2]-roi_info_aug[0], roi_info_aug[3]-roi_info_aug[1], img_mat.shape[2]]
                        roi_mat = img_mat[roi_info_aug[1]:roi_info_aug[3]+1, roi_info_aug[0]:roi_info_aug[2]+1, :]
                        roi_img_path_aug = roi_img_path.replace('.jpg', '_%s.jpg'%str(j))
                        anno_path_aug = anno_path.replace('.xml', '_%s.xml'%str(j))
                        txt_path_aug = txt_path.replace('.txt', '_%s.txt'%str(j))
                        cv2.imwrite(roi_img_path_aug, roi_mat)
                        write_XML(anno_path_aug, img_path, roi_size_aug, c, roi_info_aug, bbox_aug, y, x)
                        write_txt(txt_path_aug, img_size, roi_size_aug, c, bbox_aug, y, x, roi_info_aug)
                        cnt+=1
    print('ROI Total:%s'%str(cnt))

    return True

    
def write_XML(annoPath, imgPath, imgSize, className, roiInfo, bndBox, locationY, locationX) -> bool:
    ### create xml tree
    # root
    root = ET.Element('annotation')
    # image path
    path = ET.SubElement(root, 'path')
    # image size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    # ROI info
    roi = ET.SubElement(root, 'roi')
    roi_xmin = ET.SubElement(roi, 'xmin')
    roi_ymin = ET.SubElement(roi, 'ymin')
    roi_xmax = ET.SubElement(roi, 'xmax')
    roi_ymax = ET.SubElement(roi, 'ymax')
    # object
    obj = ET.SubElement(root, 'object')
    name = ET.SubElement(obj, 'name')
    bndbox = ET.SubElement(obj, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    ymin = ET.SubElement(bndbox, 'ymin')
    xmax = ET.SubElement(bndbox, 'xmax')
    ymax = ET.SubElement(bndbox, 'ymax')
    location = ET.SubElement(obj, 'location')
    y = ET.SubElement(location, 'y')
    x = ET.SubElement(location, 'x')
    ### value xml
    path.text = imgPath
    width.text, height.text, depth.text = str(imgSize[0]), str(imgSize[1]), str(imgSize[2])
    roi_xmin.text, roi_ymin.text, roi_xmax.text, roi_ymax.text = str(roiInfo[0]), str(roiInfo[1]), str(roiInfo[2]), str(roiInfo[3])
    name.text = className
    xmin.text, ymin.text, xmax.text, ymax.text = str(bndBox[0]), str(bndBox[1]), str(bndBox[2]), str(bndBox[3])
    y.text = str(locationY)
    x.text = str(locationX)

    ### trasfer to tree structure
    tree = ET.ElementTree(root)
    tree.write(annoPath, encoding="utf-8", xml_declaration=True)
    return True

def write_txt(txtPath, imgSize, roiSize, className, bndBox, locationY, locationX, roiInfo) -> bool:
    with open(txtPath, 'w', encoding="utf-8") as f:
        str_info = '0' # class
        x_ctr, y_ctr, bw, bh = \
            (bndBox[0]+bndBox[2])/2, \
            (bndBox[1]+bndBox[3])/2, \
            bndBox[2]-bndBox[0]+1, \
            bndBox[3]-bndBox[1]+1
        str_info = \
            str_info+' '+\
            str(x_ctr/roiSize[0])+' '+\
            str(y_ctr/roiSize[1])+' '+\
            str(bw/roiSize[0])+' '+\
            str(bh/roiSize[1]) # bbox:x,y,w,h
        str_info = str_info+' '+str(locationY)+' '+str(locationX) # location_y,location_x
        roi_x_ctr, roi_y_ctr, roi_bw, roi_bh = \
            (roiInfo[0]+roiInfo[2])/2,\
            (roiInfo[1]+roiInfo[3])/2,\
            roiInfo[2]-roiInfo[0]+1,\
            roiInfo[3]-roiInfo[1]+1
        str_info = \
            str_info+' '+\
            str(x_ctr/imgSize[1])+' '+\
            str(y_ctr/imgSize[0])+' '+\
            str(bw/imgSize[1])+' '+\
            str(bh/imgSize[0]) # roi_info:x,y,w,h
        f.write(str_info)
    # print("%s"%txtPath)
    return True

if __name__=='__main__':
    ### Globel
    Y_RANGE = [0, 80]
    X_RANGE = 8
    ### Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default='/home/huyu/dataset/kitti/training/labels', help='*.txt data')
    parser.add_argument('--classnames', default='Car,Van,Truck', help='Classes you want to crop')
    parser.add_argument('--frequency', default='2', help='Sampling frequency')
    parser.add_argument('--truncated', default='0', help='Wether crop truncated targets or not')
    parser.add_argument('--occluded', default='0', help='Wether crop occluded targets or not')
    parser.add_argument('--num', default='25000', help='How many images is the most you want to generate')
    opt = parser.parse_args()
    CLASS_NAMES = opt.classnames.split(',')
    FREQUENCY = int(opt.frequency)
    TRUNCATED = int(opt.truncated)
    OCCLUDED = int(opt.occluded)
    TARGET_NUM = int(opt.num)
    ### Crop
    crop_Kitti2ROI(opt.labels, augment=True)

     
