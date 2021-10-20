import os
from shutil import copyfile

data_root = "/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof_Croped/Data"
output_root = "data/CelebA_Spoof_Croped/Data"

def trans_data(input_file):
    with open(input_file, 'r') as f:
        for l in f:
            d = l.strip().split(',')
            filepath = os.path.join(data_root, d[0])
            if os.path.exists(filepath):
                filepath_out = os.path.join(output_root, d[0])
                out_dir, _ = os.path.split(filepath_out)
                os.makedirs(out_dir, exist_ok=True)
                copyfile(filepath, filepath_out)

if __name__ == '__main__':
    trans_data("data/high_quality_train.csv")
    trans_data("data/high_quality_test.csv")