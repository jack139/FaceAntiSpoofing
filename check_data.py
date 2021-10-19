import os
import json
from shutil import copyfile

image_root = "/media/tao/_dde_data/Datasets/CelebA_Spoof_Croped"
meta_root = image_root+"/metas/protocol2"


def check_img(filename, output_file):
    data_set = []
    for k, v in json.load(open(filename)).items():
        filepath = os.path.join(image_root, k)
        if os.path.exists(filepath):
            data_set.append("%s,%s"%(k[5:], '0' if v[-1]==1 else '1'))

    print(len(data_set))

    with open(output_file, 'w') as f:
        f.write('\n'.join(data_set))

check_img(os.path.join(meta_root, "test_on_middle_quality_device/test_label.json"), "middle_quality_test.csv")
check_img(os.path.join(meta_root, "test_on_middle_quality_device/train_label.json"), "middle_quality_train.csv")

check_img(os.path.join(meta_root, "test_on_high_quality_device/test_label.json"), "high_quality_test.csv")
check_img(os.path.join(meta_root, "test_on_high_quality_device/train_label.json"), "high_quality_train.csv")
