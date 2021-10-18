import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm import tqdm

app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))


image_root = "/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof/Data"
output_root = "/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof_Croped/Data"

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
    img_dir
     +--- id
          +--- live
          |     +--- imgs
          +--- spoof
                +--- imgs
'''
def trans_data(img_dir):
    l1 = os.listdir(os.path.join(image_root, img_dir))
    for n, i in enumerate(sorted(l1)): #id
        l2 = os.listdir(os.path.join(image_root, img_dir, i))
        for j in l2: # live, spoof
            l3 = os.listdir(os.path.join(image_root, img_dir, i, j))
            print(n, j)
            for k in tqdm(l3): # imgs and txt
                if k.endswith(".txt"): # txt 是 bbox，忽略
                    continue
                os.makedirs(os.path.join(output_root, img_dir, i, j), exist_ok=True)
                det_face(
                    os.path.join(image_root, img_dir, i, j, k),
                    os.path.join(output_root, img_dir, i, j, k),
                )


if __name__ == '__main__':
    #trans_data("train")
    trans_data("test")