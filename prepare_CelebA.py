import sys
import os

image_root = '../datasets/CelebA_Spoof_Croped/Data'

'''
    img_dir
     +--- id
          +--- live
          |     +--- imgs
          +--- spoof
                +--- imgs
'''
def trans_data(img_dir, output_file, max=10000000):
    data_set = []
    l1 = os.listdir(os.path.join(image_root, img_dir))
    print(len(l1))
    for n, i in enumerate(sorted(l1[:max])): #id
        l2 = os.listdir(os.path.join(image_root, img_dir, i))
        for j in l2: # live, spoof
            l3 = os.listdir(os.path.join(image_root, img_dir, i, j))
            for k in l3: # imgs and txt
                data_set.append("%s,%s"%(os.path.join(img_dir, i, j, k), '1' if j=="live" else '0'))

    with open(output_file, 'w') as f:
        f.write('\n'.join(data_set))

if __name__ == '__main__':
	trans_data('test', 'test.csv', 100)
	trans_data('train', 'train.csv', 400)
