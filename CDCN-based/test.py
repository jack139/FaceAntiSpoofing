import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
from utils.utils import read_cfg, build_network
from utils.eval import predict
from datetime import datetime
from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))


cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = build_network(cfg)


val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

saved_name = os.path.join(cfg['output_dir'], "CDCNpp_nuaa_e4_acc_0.8780.pth")
state = torch.load(saved_name, map_location=device)
network.load_state_dict(state['state_dict'])
print("load model: ", saved_name)

if device.type!='cpu':
    network.cuda()


def test(img):
    network.eval()

    img = val_transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        img = img.to(device)
        net_depth_map, _, _, _, _, _ = network(img)

        preds, score = predict(net_depth_map)

    return preds, score

if __name__ == '__main__':
    
    if len(sys.argv)<2:
        print("usage: python3 %s <image_path>" % sys.argv[0])
        sys.exit(1)

    frame = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

    faces = app.get(frame, max_num=100) # 检测人脸


    for face in faces:
        sub_img = face_align.norm_crop(frame, landmark=face.kps, image_size=256) # 人脸修正
        #cv2.imwrite('img_test.jpg',sub_img)

        face_img = Image.fromarray(sub_img[:, :, ::-1])
        #face_img.save('img2_test.jpg')

        start_time = datetime.now()
        preds, score = test(face_img)
        print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        print(preds)
        print(score)
