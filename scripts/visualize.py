import os
# import shutil
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from math import ceil

############################ManFu XML################################
# <?xml version="1.0" encoding="utf-8"?>

# <annotation>
#   <folder>Desktop</folder>
#   <filename>09-30-16-173_38.jpg</filename>
#   <source>
#     <database>My Database</database>
#   </source>
#   <size>
#     <depth>3</depth>
#   </size>
#   <segmented>0</segmented>
#   <object>
#     <name>car</name>
#     <pose>Unspecified</pose>
#     <truncated>0</truncated>
#     <difficult>0</difficult>
#     <bndbox>
#       <xmin>17</xmin>
#       <ymin>13</ymin>
#       <xmax>82</xmax>
#       <ymax>61</ymax>
#     </bndbox>
#   </object>
# </annotation>
######################################################################

#############################Anngic XML###############################
# <?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
# <annotation>
#     <path>.\ROI_20210223152609\roi_annotaions\15-26-21-963_2980.xml</path>
#     <size>
#         <width>132</width>
#         <height>88</height>
#         <depth>3</depth>
#     </size>
#     <roi>
#         <xmin>410</xmin>
#         <ymin>484</ymin>
#         <xmax>542</xmax>
#         <ymax>572</ymax>
#     </roi>
#     <object>
#         <name>car</name>
#         <bndbox>
#             <xmin>22</xmin>
#             <ymin>16</ymin>
#             <xmax>520</xmax>
#             <ymax>559</ymax>
#         </bndbox>
#         <location>
#             <y>31.200003</y>
#             <x>-4.078982</x>
#         </location>
#     </object>
# </annotation>
######################################################################


def VisualizeManfu(folder):
    img_folder = os.path.join(folder, "images")
    xml_folder = os.path.join(folder, "xmls")
    vis_folder = os.path.join(folder, "visualized")
    os.removedirs(vis_folder)
    os.mkdir(vis_folder)

    file_list = os.listdir(xml_folder) 

    for f in tqdm(file_list):
        file_id = f.split(".")[0]

        tree = ET.parse(os.path.join(xml_folder, f))
        root = tree.getroot()
        name = root.find("object").find("name").text
        bndbox = root.find("object").find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        img_path = os.path.join(img_folder, file_id+".jpg")
        img = cv2.imread(img_path)
        if name=="car":
            color = (0,255,0)
        elif name=="bus":
            color = (255,0,0)
        elif name=="truck":
            color = (0,0,255)
        elif name=="tricycle":
            color = (0,196,196)
        elif name=="person":
            color = (196,196,0)
        elif name=="person":
            color = (196,0,196)
        else:
            color = (255,255,255)

        thickness = img.shape[0]/128 if img.shape[0]<img.shape[1] else img.shape[1]/128 
        img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 2)
        cv2.imwrite(os.path.join(vis_folder, file_id+".jpg"), img)

    return


def VisualizeAnngic(folder):
    img_folder = os.path.join(folder, "images")
    xml_folder = os.path.join(folder, "annotations")
    vis_folder = os.path.join(folder, "visualized")
    if os.path.exists(vis_folder):
        os.removedirs(vis_folder)
    os.mkdir(vis_folder)

    file_list = os.listdir(xml_folder) 

    for f in tqdm(file_list):
        file_id = f.split(".")[0]

        tree = ET.parse(os.path.join(xml_folder, f))
        root = tree.getroot()
        name = root.find("object").find("name").text
        bndbox = root.find("object").find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        img_path = os.path.join(img_folder, file_id+".jpg")
        img = cv2.imread(img_path)
        if name=="car":
            color = (0,255,0)
        elif name=="bus":
            color = (255,0,0)
        elif name=="truck":
            color = (0,0,255)
        elif name=="tricycle":
            color = (0,196,196)
        elif name=="person":
            color = (196,196,0)
        elif name=="person":
            color = (196,0,196)
        else:
            color = (255,255,255)

        thickness = img.shape[0]/128 if img.shape[0]<img.shape[1] else img.shape[1]/128 
        img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 2)
        cv2.imwrite(os.path.join(vis_folder, file_id+".jpg"), img)

    return

if __name__ == "__main__":
    # VisualizeManfu("/home/huyu/ROI")

    roots = ["/home/huyu/dataset/fv1xm/roi/train", "/home/huyu/dataset/fv1xm/roi/test"]
    classes = ["car", "bus", "truck", "tricycle", "person", "rider"]
    for r in roots:
        for c in classes:
            folder = os.path.join(r, c)
            VisualizeAnngic(folder)


