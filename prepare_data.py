import os
from shutil import copyfile

train_csv = "../CDCN-Face-Anti-Spoofing.pytorch/data/train.csv"
val_csv = "../CDCN-Face-Anti-Spoofing.pytorch/data/val.csv"

data_root = "../../datasets/NUAA_Detectedface"
output_data = "data/dataset"

def trans_data(input_file, output_dir):
    with open(input_file, 'r') as f:
        for l in f:
            d = l.strip().split(',')
            _, img_file = os.path.split(d[0])
            copyfile(os.path.join(data_root, d[0]), os.path.join(output_dir, d[1], img_file))

if __name__ == '__main__':
    train_dir = os.path.join(output_data, "train")
    val_dir = os.path.join(output_data, "val")
    os.makedirs('%s/1'%train_dir, exist_ok=True)
    os.makedirs('%s/0'%train_dir, exist_ok=True)
    os.makedirs('%s/1'%val_dir, exist_ok=True)
    os.makedirs('%s/0'%val_dir, exist_ok=True)

    trans_data(train_csv, train_dir)
    trans_data(val_csv, val_dir)