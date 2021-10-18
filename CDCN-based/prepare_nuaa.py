import sys
import os

data_root = '../datasets/NUAA_Detectedface'
true_train_file = 'client_train_face.txt'
true_test_file = 'client_test_face.txt'
fake_train_file = 'imposter_train_face.txt'
fake_test_file = 'imposter_test_face.txt'
output_train = 'data/train.csv'
output_test = 'data/val.csv'

def load_txt(filename, label, prefix):
    data = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            d = l.split()[0].split('\\')
            data.append("%s,%s"%(os.path.join(prefix, d[0], d[1]), label))
    return data

true_train = load_txt(os.path.join(data_root, true_train_file), '1', 'ClientFace')
true_test = load_txt(os.path.join(data_root, true_test_file), '1', 'ClientFace')
fake_train = load_txt(os.path.join(data_root, fake_train_file), '0', 'ImposterFace')
fake_test = load_txt(os.path.join(data_root, fake_test_file), '0', 'ImposterFace')

train_set = true_train + fake_train
test_set = true_test + fake_test

with open(output_train, 'w') as f:
    f.write('\n'.join(train_set))

with open(output_test, 'w') as f:
    f.write('\n'.join(test_set))
