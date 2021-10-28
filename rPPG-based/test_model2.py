import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import cv2
#########################
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
#########################
from rPPG2.rPPG_Extracter import *
#########################
from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(224, 224))


# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")
print("[INFO] Model is loaded from disk")


dim = (128,128)
def get_rppg_pred(frame):
    use_classifier = True  # Toggles skin classifier
    sub_roi = []           # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]

    rPPG_extracter = rPPG_Extracter()
        
    rPPG = []

    rPPG_extracter.measure_rPPG(frame,use_classifier,sub_roi) 
    rPPG = np.transpose(rPPG_extracter.rPPG)

    return rPPG
    

def make_pred(li):
    [single_img,rppg] = li
    single_img = cv2.resize(single_img, dim)
    single_x = img_to_array(single_img)
    single_x = np.expand_dims(single_x, axis=0)
    single_pred = model.predict([single_x,rppg])
    return single_pred


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret:        
        faces = app.get(frame, max_num=100) # 检测人脸

        # Draw a rectangle around the faces
        rimg = app.draw_on(frame, faces)

        #for (top, right, bottom, left) in faces:
        for face in faces:
            sub_img = face_align.norm_crop(frame, landmark=face.kps, image_size=256) # 人脸修正
            rppg_s = get_rppg_pred(sub_img)
            rppg_s = rppg_s.T

            pred = make_pred([sub_img,rppg_s])

            cv2.putText(rimg,"Real: "+str(pred[0][0]), (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.putText(rimg,"Fake: "+str(pred[0][1]), (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('To quit press q', rimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
