import json
import cv2
import numpy as np
import os
import tqdm
from pathlib import Path
import shutil
def build_image_with_label(img,config):
    img_shape = (img_config['width'],img_config['height'])
    img = cv2.resize(img, img_shape)
    boxes = config['boxes']
    for ind,(box,object) in enumerate(zip(boxes,img_config['names'])):
        x1,y1,x2,y2,score = box
        img = cv2.rectangle(img,(int(x1),int(y1)), (int(x2), int(y2)),(0,0,255), 1)
        cv2.putText(img,object+f"_{ind}",(int(x1),int(y1)-5),cv2.FONT_HERSHEY_COMPLEX ,1, (0,0,255), 1)
    return img

# image_dir = "C:\\Users\\bororo\Pictures\movieclips_Knowing"
raw_dir ="//nas-alinlp/lcl193798/data_renm/VCR/vcr1images"
target_dir = "//nas-alinlp/lcl193798/data_renm/VCR/vcr1images_label"


for file in tqdm.tqdm(os.listdir(raw_dir)):
    if file.endswith(".txt"):
        continue
    img_dir = os.path.join(raw_dir,file)
    tgt_dir = img_dir.replace(raw_dir, target_dir)
    print(f"creating labeled image dir: {tgt_dir}")
    Path(tgt_dir).mkdir(parents=True, exist_ok=True)
    for image_file in tqdm.tqdm(os.listdir(img_dir)):

        if image_file.endswith(".jpg"):
            img_config_name = image_file.replace("jpg","json")
            img = cv2.imread(os.path.join(img_dir, image_file))
            img_config = json.load(open(os.path.join(img_dir, img_config_name)))
            img_with_label = build_image_with_label(img,img_config)
            cv2.imwrite(os.path.join(tgt_dir, image_file), img)
            shutil.copyfile(os.path.join(img_dir, img_config_name),os.path.join(tgt_dir, img_config_name))
        else:
            continue


 