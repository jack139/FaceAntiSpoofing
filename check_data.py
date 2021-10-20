import os
import json, random
from shutil import copyfile
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm import tqdm

image_root = "/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof"
#output_root = "/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof_Croped"
output_root = "/media/tao/_dde_data/Datasets/CelebA_Spoof_Croped"
meta_root = output_root+"/metas/protocol2"

output_size = 256

def det_face(input_file, output_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    faces = app.get(img, max_num=1) # 检测人脸
    if len(faces)==0:
        print(input_file)
        return
    aimg = face_align.norm_crop(img, landmark=faces[0].kps, image_size=output_size) # 人脸修正
    cv2.imwrite(output_file, aimg)

'''
                    0       1       2       3       4       5           6       7   8   9       10
40 Spoof Type       Live    Photo   Poster  A4      Face    Upper       Region  PC  Pad Phone   3D 
                                                    Mask    Body Mask   Mask                    Mask
41 Illumination 
   Condition        Live    Normal  Strong  Back    Dark                        
42 Environment      Live    Indoor  Ourdoor                             
43 Live/Spoof       Live    Spoof
   label 
'''

spoof_type_filter = [1, 3, 4, 5, 7, 8, 9]

# 生成 训练用 csv
def check_img(filename, output_file):
    data_set = []
    li = sp = 0
    for k, v in json.load(open(filename)).items():
        filepath = os.path.join(output_root, k)
        if os.path.exists(filepath):
            if v[43]==0: # live
                data_set.append("%s,%s"%(k[5:], '1'))
                li += 1
                pass
            else: # spoof
                if v[40] in spoof_type_filter:
                    data_set.append("%s,%s"%(k[5:], '0'))
                    sp += 1

    print(len(data_set), li, sp)

    with open(output_file, 'w') as f:
        f.write('\n'.join(data_set))


# 从原始图，生成人脸图
def trans_to_face(filename):
    data_set = []
    li = sp = 0
    for k, v in tqdm(json.load(open(filename)).items()):
        filepath = os.path.join(image_root, k)
        filepath_out = os.path.join(output_root, k)
        if os.path.exists(filepath_out): # 已存在
            continue
        if (v[43]==1) and (v[40] not in spoof_type_filter): # 不符合要求
            continue

        out_dir, _ = os.path.split(filepath_out)
        os.makedirs(out_dir, exist_ok=True)
        det_face(filepath, filepath_out)


# 从json文件 按类别和百分比 生成新的文件
def trans_to_json(filename, filename_out, ratio):
    w = json.load(open(filename)).items()
    total = len(w)

    live = []
    spoof = {}
    for x in spoof_type_filter:
        spoof[x] = []

    print("total= %d\tratio= %.4f"%(total, ratio))

    for k, v in w:
        if v[43]==0: # live
            live.append((k, v))
        else: # spoof
            if v[40] in spoof_type_filter:
                spoof[v[40]].append((k,v))

    new_set = []
    random.shuffle(live)
    new_set.extend(live[:int(len(live)*ratio)])
    print("live: %d --> %d"%(len(live), len(new_set)))

    for k in spoof.keys():
        random.shuffle(spoof[k])
        tt = spoof[k][:int(len(spoof[k])*ratio)]
        new_set.extend(tt)
        print("spoof[%d]: %d --> %d"%(k, len(spoof[k]), len(tt)))

    json.dump(dict(new_set), open(filename_out, 'w'))

    print("new_set= ", len(new_set))


if __name__ == '__main__':

    # 按比例生成指定数量的json
    trans_to_json(meta_root+"/test_on_high_quality_device/test_label.json", "test_label.json", 0.1)
    trans_to_json(meta_root+"/test_on_high_quality_device/train_label.json", "train_label.json", 0.06)

    # 生成人脸图片
    #trans_to_face("test_label.json")
    #trans_to_face("train_label.json")

    # 生成训练用 csv
    #check_img("test_label.json", "high_quality_test.csv")
    #check_img("train_label.json", "high_quality_train.csv")

    