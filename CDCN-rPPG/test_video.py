import sys
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from utils.utils import read_cfg, build_network
from utils.eval import predict
from datetime import datetime
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from datasets.FASDataset import get_rppg_pred

PREDICT_THRESHOLD = 0.26
USE_MIX = True

app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(224, 224))


cfg = read_cfg(cfg_file="config/CDCN_adam_lr1e-3.yaml")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = build_network(cfg)


val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

saved_name = os.path.join(cfg['output_dir'], "CDCN_CelebA_Spoof_e12_acc_0.9441.pth")
state = torch.load(saved_name, map_location=device)
network.load_state_dict(state['state_dict'])
print("load model: ", saved_name)

if device.type!='cpu':
    network.cuda()


def test(img, rppg_s, mix=True):
    network.eval()

    img = val_transform(img)
    img = img.unsqueeze(0)

    rppg_s = torch.from_numpy(rppg_s.astype(np.float32))
    rppg_s = rppg_s.unsqueeze(0)

    with torch.no_grad():
        img = img.to(device)
        rppg_s = rppg_s.to(device)
        rppg_depth, net_depth_map, _, _, _, _, _ = network(img, rppg_s)

        if mix:
            mix_depth = (net_depth_map + rppg_depth) / 2 # 算术平均
        else:
            mix_depth = net_depth_map
        preds, score = predict(mix_depth, threshold=PREDICT_THRESHOLD)

    return preds, score


if __name__ == '__main__':

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret:
            faces = app.get(frame, max_num=100) # 检测人脸

            # Draw a rectangle around the faces
            rimg = app.draw_on(frame, faces)
            
            for face in faces:

                sub_img = face_align.norm_crop(frame, landmark=face.kps, image_size=256) # 人脸修正
                #cv2.imwrite('img_test.jpg',sub_img)

                # 取得 rPPG 特征
                rppg_s = get_rppg_pred(sub_img)
                rppg_s = rppg_s.T[0]

                face_img = Image.fromarray(sub_img[:, :, ::-1])
                #face_img.save('img2_test.jpg')

                preds, score = test(face_img, rppg_s, USE_MIX)



                cv2.putText(rimg, "Real: "+str(score), (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('To quit press q', rimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
