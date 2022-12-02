import os
import shutil
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

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

def Manfu2Anngic(folder):
    agc_folder = os.path.join(folder, "annotations")
    manfu_folder = os.path.join(folder, "thirdparty-xmls")

    file_list = os.listdir(agc_folder) 


    for f in tqdm(file_list):
        # file_id = f.split(".")[0]
        
        manfu_tree = ET.parse(os.path.join(manfu_folder, f))
        manfu_root = manfu_tree.getroot()
        manfu_name = manfu_root.find("object").find("name").text
        manfu_pose = manfu_root.find("object").find("pose").text
        manfu_difficult = manfu_root.find("object").find("difficult").text
        manfu_truncated = manfu_root.find("object").find("truncated").text
        manfu_bbox = manfu_root.find("object").find("bndbox")
        manfu_xmin = manfu_bbox.find("xmin").text
        manfu_ymin = manfu_bbox.find("ymin").text
        manfu_xmax = manfu_bbox.find("xmax").text
        manfu_ymax = manfu_bbox.find("ymax").text

        agc_tree = ET.parse(os.path.join(agc_folder, f))
        agc_root = agc_tree.getroot()
        agc_obj = agc_root.find("object")
        agc_obj.find("name").text = manfu_name
        try:
            agc_obj.find("pose").text = manfu_pose
        except:
            agc_obj.append(ET.Element("pose"))
            agc_obj.find("pose").text = manfu_pose
        try:
            agc_obj.find("difficult").text = manfu_difficult
        except:
            agc_obj.append(ET.Element("difficult"))
            agc_obj.find("difficult").text = manfu_difficult
        try:
            agc_obj.find("truncated").text = manfu_truncated
        except:
            agc_obj.append(ET.Element("truncated"))
            agc_obj.find("truncated").text = manfu_truncated
        agc_bbox = agc_root.find("object").find("bndbox")
        agc_bbox.find("xmin").text = manfu_xmin
        agc_bbox.find("ymin").text = manfu_ymin
        agc_bbox.find("xmax").text = manfu_xmax
        agc_bbox.find("ymax").text = manfu_ymax

        agc_tree.write(os.path.join(agc_folder, f))
    return



if __name__ == "__main__":
    roots = ["/home/huyu/dataset/fv1xm/roi-gen2/train", "/home/huyu/dataset/fv1xm/roi-gen2/test"]
    classes = ["person", "car", "bus", "truck", "tricycle", "rider"]
    for r in roots:
        for c in classes:
            folder = os.path.join(r, c)
            Manfu2Anngic(folder)

