
import os
import shutil
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

######################################################################
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


def DistributeClasses(src, dst):
    # src_img_folder = os.path.join(src, "images")
    src_xml_folder = os.path.join(src, "vocxml")

    file_list = os.listdir(src_xml_folder) 

    # dst_anno_folder = os.path.join(dst, "roi_annotations")
    # anno_files = os.listdir(dst_anno_folder)
    # anno_ids = [f.split(".")[0] for f in anno_files]


    for f in tqdm(file_list):
        file_id = f.split(".")[0]

        # if file_id not in anno_ids:
        #     continue
        
        tree = ET.parse(os.path.join(src_xml_folder, f))
        root = tree.getroot()
        name = root.find("object").find("name").text
        if (name not in ["bus", "truck"]): print(f)

        # src_img_dir = os.path.join(src_img_folder, file_id+".jpg")
        src_xml_dir = os.path.join(src_xml_folder, file_id+".xml")
        # dst_img_dir = os.path.join(dst, name, "images", file_id+".jpg")
        dst_xml_dir = os.path.join(dst, name, "thirdparty-xmls", file_id+".xml")
        if not os.path.isdir(os.path.join(dst, name, "thirdparty-xmls")):
            os.mkdir(os.path.join(dst, name, "thirdparty-xmls"))
        # shutil.copyfile(src_img_dir ,dst_img_dir)
        shutil.copyfile(src_xml_dir ,dst_xml_dir)

    return



if __name__ == "__main__":
    # DistributeClasses("/home/huyu/dataset/fv1xm/roi-gen2/RawLabel/train", "/home/huyu/dataset/fv1xm/roi-gen2/train")
    # DistributeClasses("/home/huyu/dataset/fv1xm/roi-gen2/RawLabel/test", "/home/huyu/dataset/fv1xm/roi-gen2/test")

    # DistributeClasses("/home/huyu/dataset/fv1xm/roi-gen2/RawLabel-bus/train", "/home/huyu/dataset/fv1xm/roi-gen2/train")

    DistributeClasses("/home/huyu/dataset/fv1xm/roi-gen2/RawLabel-truck/train", "/home/huyu/dataset/fv1xm/roi-gen2/train")