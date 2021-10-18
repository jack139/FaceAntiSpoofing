import os
from test_img import fas_check2

nuaa_root = "/media/tao/_dde_data/Datasets/NUAA_Detectedface"

train_csv = "train.csv"
val_csv = "val.csv"

def test(input_file):
    N = 0
    T = 0
    with open(input_file, 'r') as f:
        for l in f:
            d = l.strip().split(',')
            _, img_file = os.path.split(d[0])
            img_path = os.path.join(nuaa_root, d[0])
            r = fas_check2(img_path)
            N += 1
            if round(r[0])==int(d[1]):
                T += 1

            #print(img_file, r[0], d[1], round(r[0])==int(d[1]))

    return T/N

# rPPG on NUAA:
# train 0.7244
# val 0.5801

if __name__ == '__main__':
    acc1 = test(os.path.join(nuaa_root, train_csv))
    print(acc1)

    acc2 = test(os.path.join(nuaa_root, val_csv))
    print(acc1, acc2)
