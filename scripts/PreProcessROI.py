import os
from tqdm import tqdm
from xml.etree import ElementTree as ET
import shutil


def ReadYLocation(xmlDir:str):
    tree = ET.parse(xmlDir)
    root = tree.getroot()
    obj = root.find("object")
    loc = obj.find("location")
    y_loc = loc.find("y")
    return float(y_loc.text)


def ReadXLocation(xmlDir:str):
    tree = ET.parse(xmlDir)
    root = tree.getroot()
    obj = root.find("object")
    loc = obj.find("location")
    x_loc = loc.find("x")
    return float(x_loc.text)

def FilterFv1xmROI(folder:str):
    try:
        img_folder = os.path.join(folder, "roi_images")
        xml_folder = os.path.join(folder, "roi_annotations")
    except:
        Exception

    img_files = os.listdir(img_folder)
    xml_files = os.listdir(xml_folder)
    # xml_files = list([xml_dir for xml_dir in xml_files]) #for tqdm

    # for xml_dir in tqdm(xml_files):
    for xml_file in tqdm(xml_files):
        may_be_img_file = xml_file.replace("xml", "jpg")
        if may_be_img_file not in img_files:
            xml_dir = os.path.join(xml_folder, xml_file)
            os.remove(xml_dir) 
            # print(xml_dir)
            continue
        x_loc = ReadXLocation(os.path.join(xml_folder, xml_file))
        y_loc = ReadYLocation(os.path.join(xml_folder, xml_file))
        if (not -8<=x_loc<=8) or (not 0<=y_loc<=100):
            img_dir = os.path.join(img_folder, may_be_img_file)
            xml_dir = os.path.join(xml_folder, xml_file)
            os.remove(img_dir)
            os.remove(xml_dir) 
            # print("remove:", img_dir, "&&", xml_dir)


def ReadROIXMLInfo(xmlDir:str):
    tree = ET.parse(xmlDir) 
    root = tree.getroot()
    file_id = xmlDir.split("/")[-1].split(".")[0]
    size = root.find("size")
    roi_size = [size.find("width").text, size.find("height").text]
    roi_box = [root.find("roi").find("xmin").text, root.find("roi").find("ymin").text,
                root.find("roi").find("xmax").text, root.find("roi").find("ymax").text]
    obj = root.find("object")
    obj_id = CLASS[obj.find("name").text]
    obj_box = [obj.find("bndbox").find("xmin").text, obj.find("bndbox").find("ymin").text,
                obj.find("bndbox").find("xmax").text, obj.find("bndbox").find("ymax").text]
    location_Y = obj.find("location").find("y").text
    location_X = obj.find("location").find("x").text

    return file_id, roi_size, obj_id, obj_box, location_Y, location_X, roi_box



def WriteTXT(txtPath, imgSize, roiSize, classID, bndBox, locationY, locationX, roiInfo) -> bool:
    with open(txtPath, 'w', encoding="utf-8") as f:
        str_info = classID # class
        x_ctr, y_ctr, bw, bh = (float(bndBox[0])+float(bndBox[2]))/2, (float(bndBox[1])+float(bndBox[3]))/2, float(bndBox[2])-float(bndBox[0])+1, float(bndBox[3])-float(bndBox[1])+1
        str_info = str(str_info)+' '+str(x_ctr/float(roiSize[0]))+' '+str(y_ctr/float(roiSize[1])) +' '+str(bw/float(roiSize[0]))+' '+str(bh/float(roiSize[1])) # bbox:x,y,w,h
        # str_info = str_info+' '+str(float(locationY)/100)+' '+str((float(locationX)+8)/16) # location_y,location_x
        # roi_x_ctr, roi_y_ctr, roi_bw, roi_bh = (float(roiInfo[0])+float(roiInfo[2]))/2, (float(roiInfo[1])+float(roiInfo[3]))/2, float(roiInfo[2])-float(roiInfo[0])+1, float(roiInfo[3])-float(roiInfo[1])+1
        # str_info = str_info+' '+str(roi_x_ctr/imgSize[0])+' '+str(roi_y_ctr/imgSize[1])+' '+str(roi_bw/imgSize[0])+' '+str(roi_bh/imgSize[1]) # roi_info:x,y,w,h
        f.write(str_info)
    # print("%s"%txtPath)
    return True

def GenerateTXT(folder:str):
    try:
        xml_folder = os.path.join(folder, "annotations")
        txt_folder = os.path.join(folder, "labels")
        if os.path.exists(txt_folder):
            shutil.rmtree(txt_folder)
        os.mkdir(txt_folder)
    except:
        Exception
    xml_files = os.listdir(xml_folder)
    for file in tqdm(xml_files):
        xml_dir = os.path.join(xml_folder, file)
        file_id, roi_size, obj_id, obj_box, location_Y, location_X, roi_box = ReadROIXMLInfo(xml_dir)
        txt_file = file.replace("xml", "txt")
        txt_dir = os.path.join(txt_folder, txt_file)
        WriteTXT(txt_dir, IMAGE_SIZE, roi_size, obj_id, obj_box, location_Y, location_X, roi_box)


if __name__ == "__main__":
    IMAGE_SIZE = [1280, 1080]
    CLASS = {"person":0, "car":1, "bus":2, "truck":3, "tricycle":4, "rider":5}

    # root = "/home/huyu/dataset/fv1xm/roi"
    # for folder in os.listdir(root):
    #     FilterFv1xmROI(os.path.join(root, folder))
    roots = ["/home/huyu/dataset/fv1xm/roi-gen2/train", "/home/huyu/dataset/fv1xm/roi-gen2/test"]
    # roots = ["/home/huyu/dataset/fv1xm/roi-boyue-test-tmp"]
    classes = ["person", "car", "bus", "truck", "tricycle", "rider"]
    for r in roots:
        for c in classes:
            data_folder = os.path.join(r, c)
            GenerateTXT(data_folder)
    
    # FilterFv1xmROI("/home/huyu/dataset/fv1xm/roi-gen2/ROI_20210318144436")
    # FilterFv1xmROI("/home/huyu/dataset/fv1xm/roi-gen2/ROI_20210318164422")
    # FilterFv1xmROI("/home/huyu/dataset/fv1xm/roi-gen2/ROI_20210322155010")
    # FilterFv1xmROI("/home/huyu/dataset/fv1xm/roi-gen2/ROI_20210322163633_0")
    # FilterFv1xmROI("/home/huyu/dataset/fv1xm/roi-gen2/ROI_20210322163633_1")