'''
Author: huyu
Date: 2020-12-01 10:03:53
LastEditTime: 2020-12-07 15:47:39
LastEditors: Please set LastEditors
Description: Gennerate images list txt in darknet format.
FilePath: /yolov3-roidepth/scripts/GenerateDatalist.py
'''

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

'''
description: generate images paths in *.txt, visualize labels and depth distribution.
'''
def GenerateDatalist(folder:str, classes:list, suffix:str, visualize=True) -> bool:
    txt_dir = os.path.join(folder, "fileList.txt")
    txt_file = open(txt_dir, 'w', encoding='utf-8')
    for c in classes:
        print("Generate list of %s in %s:"%(c, folder))
        txtid_dir = os.path.join(folder, c, "fileList_ids.txt")
        txtid_file = open(txtid_dir , 'w', encoding='utf-8')
        class_folder = os.path.join(folder, c)
        # class_folder = folder
        img_folder = os.path.join(class_folder, "images")
        # img_folder = os.path.join(class_folder, "JPEGImages")
        assert os.path.isdir(img_folder), "Path %s is not exist."%img_folder
        assert os.path.isdir(img_folder.replace('images','labels')), "You need to put 'images' and 'labels' in a same folder." 
        img_files = os.listdir(img_folder)
        img_ids = [imgfile.split(".")[0]+"\n" for imgfile in img_files]
        img_files = [os.path.join(img_folder, f+'\n') for f in img_files]
        # generate txt in darknet format.
        # if c == "car":
        #     txt_file.writelines(img_files[0:8000])
        #     txtid_file.writelines(img_ids[0:8000])
        # else:
        txt_file.writelines(img_files)
        txtid_file.writelines(img_ids)
        txtid_file.close()


        # visualize labels if visualize=True
        x_lt, y_lt = [], []
        if visualize:
            visualFolder = img_folder.replace('images','visualized')
            if not os.path.exists(visualFolder):
                os.mkdir(visualFolder)
            # go over image paths.
            for f in tqdm(img_files):
                if c == "bg": continue
                f = f.strip('\n')
                img_id = f.split('/')[-1].split('.')[0]
                img_mat = cv2.imread(f)
                label_file = f.replace('images','labels').replace('.%s'%suffix,'.txt')
                with open(label_file, 'r', encoding='utf-8') as lf:
                    label_lt = lf.readlines()
                label_lt = [lt.split(' ') for lt in label_lt]
                # do visualization
                VisualizeLabels(visualFolder, img_id, img_mat, label_lt)
                # save stats for plot
                x_lt.extend(np.asarray(label_lt)[:,6])
                y_lt.extend(np.asarray(label_lt)[:,5])
            if c != "bg":
                x_lt, y_lt = [float(xi)*16-8 for xi in x_lt], [float(yi)*Y_RANGE for yi in y_lt]
                # do plot
                plot_file = os.path.join(class_folder, "Y-distribution.jpg")
                try:
                    plot_xy(x_lt, y_lt, plot_file)
                except:
                    Exception

    txt_file.close()

    return True

'''
description: visualize class, bounding box and depth(if exists).
'''
def VisualizeLabels(saveFolder:str, imgID:str, imgMat:np.array, labelList:list) -> bool:
    # set box colors and classes.
    colors = [(0,0,191),(0,191,0),(0,191,191),(191,0,0),(240,22,160),(191,191,0)]
    classes = ['person', 'car', 'bus' ,'truck', 'tricycle', 'rider']
    # get image shape.
    img_h, img_w, _ = imgMat.shape
    # go over labels.
    for l in labelList:
        # get class,x,y,w,h.
        c, x, y, w, h = l[0:5]
        c, x, y, w, h = int(c), float(x)*img_w, float(y)*img_h, float(w)*img_w, float(h)*img_h
        xmin, ymin, xmax, ymax = round(x-w/2), round(y-h/2), round(x+w/2), round(y+h/2)
        # draw bbox.
        imgMat = cv2.rectangle(imgMat, (xmin, ymin), (xmax, ymax), colors[c], int(min(img_w,img_h)/128))
        # if num of label data > 5, it mean depth is labeled, get depth data and visualize class and depth, else visualize class only. 
        c_text = classes[c]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 2
        if len(l)>5:
            d = float(l[5])*Y_RANGE
            # visualize class and depth.
            cv2.putText(imgMat, '%s Y=%.1f'%(c_text, d), (xmin, ymin-font_size), font, min(img_w,img_h)/256, colors[c], int(min(img_w,img_h)/128))
        else:
            # visualize class only.
            cv2.putText(imgMat, '%s'%c_text, (xmin, ymin-font_size), font, min(img_w,img_h)/256, colors[c], int(min(img_w,img_h)/64))
    # save final image
    savePath = os.path.join(saveFolder, '%s.jpg'%imgID)
    cv2.imwrite(savePath, imgMat)
    return True

'''
description: Plot x and y distribution, and save as images.
'''
def plot_xy(xList:list, yList:list, savePath:str, save=True) -> bool:
    plt.plot(xList, yList, 'ro')
    plt.xlabel('X in meters')
    plt.ylabel('Y in meters')
    x_max, y_max = max(max(xList),abs(min(xList))), max(yList)
    x_axis, y_axis = int(x_max*1.2), int(y_max*1.2)
    plt.axis([-x_axis, x_axis, 0 ,y_axis])
    if save:
        plt.savefig(savePath)
    plt.show()
    return True


if __name__=='__main__':
    # Y_RANGE = 100
    # roots = ["/home/huyu/dataset/fv1xm/roi/train", "/home/huyu/dataset/fv1xm/roi/test"]
    # Y_RANGE = 80
    # roots = ["/home/huyu/dataset/AGC-FVM/roidepth/kitti/train", "/home/huyu/dataset/AGC-FVM/roidepth/kitti/test"]
    # Y_RANGE = 100
    # roots = [
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/care_person1/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/care_person2/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/care_person3/daytime/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/care_person3/night/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/care_person4/daytime/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/care_person4/night/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/negaSample1/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/negaSample2/daytime/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/negaSample2/night/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/train/night/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/train/rainy/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/train/sunny/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/train/very_sunny/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/yutong/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fv10/1/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fv10/2/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fake_car/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fake_person_ps/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fake_person_2006/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fake_car_200704/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/fake_person_200704/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/background_200704/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake/fake_car/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake/fake_person/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake/background/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake_2/background/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake_2/background2/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake_2/fake_person/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake3/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake4/background/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageOne/aeb_data_fake4/fake_car_person/',

    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageZero/train/night/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageZero/train/sunny/',

    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageTwo/train/sunny/1/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageTwo/train/sunny/2/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageTwo/train/sunny/3/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageTwo/train/sunny/4/',
    #         '/home/huyu/project/AGC-FVM/darknet/dongfeng/stageTwo/train/sunny/5/'
    #         ]
    Y_RANGE = 100
    # roots = ["/home/huyu/dataset/fv1xm/roi-gen2/train", "/home/huyu/dataset/fv1xm/roi-gen2/test"]
    roots = ["/home/huyu/dataset/fv1xm/roi-gen2/test"]
    # roots = ["/home/huyu/dataset/fv1xm/roi-boyue-test-tmp"]
    suffix = "jpg"
    # classes = ["car", "bus", "truck", "bg"]
    classes = ["car", "bus", "truck"]
    visualize_flag = False
    for r in roots:
        GenerateDatalist(r, classes, suffix, visualize=visualize_flag)
