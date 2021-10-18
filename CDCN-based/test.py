import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
from utils.utils import read_cfg, build_network
from utils.eval import predict
import face_recognition
from datetime import datetime

cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = build_network(cfg)


val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

saved_name = os.path.join(cfg['output_dir'], "CDCNpp_nuaa_e16_acc_0.8658.pth")
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

    img_name = sys.argv[1]

    #img = Image.open(img_name)
    #test(img)

    frame = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    faces = face_recognition.face_locations(frame)

    for (top, right, bottom, left) in faces:
        x, y, w, h = left, top, right-left+1, bottom-top+1
        sub_img=frame[y:y+h,x:x+w]

        #cv2.imwrite('img_%d_%d.jpg'%(x,y),sub_img)

        face_img = Image.fromarray(sub_img[:, :, ::-1])

        #face_img.save('img2_%d_%d.jpg'%(x,y))

        start_time = datetime.now()
        preds, score = test(face_img)
        print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        print(preds)
        print(score)
