# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam, SGD

#########################
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
#########################
from rPPG2.rPPG_Extracter import *


batch_size = 128
steps_per_epoch = 100
epochs = 5
learning_rate=1e-3
train_dir = 'data/dataset/val'
val_dir = 'data/dataset/train'


###### rPPG 
dim = (128,128)
def get_rppg_pred(frame):
    use_classifier = True  # Toggles skin classifier
    sub_roi = []           # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]

    rPPG_extracter = rPPG_Extracter()
        
    rPPG = []

    rPPG_extracter.measure_rPPG(frame,use_classifier,sub_roi) 
    rPPG = np.transpose(rPPG_extracter.rPPG)

    return rPPG
   

# 数据生成器
def data_generator(data_path, batch_size):
    file_list = []
    for i in os.listdir(os.path.join(data_path, '0')):
        file_list.append((i, '0'))
    for i in os.listdir(os.path.join(data_path, '1')):
        file_list.append((i, '1'))
    random.shuffle(file_list)

    print(data_path, ": ", len(file_list), "\tbatch: ", batch_size)

    while True:
        for n in range(len(file_list)//batch_size):
            X1 = X2 = y = None
            for m in range(batch_size):
                i = file_list[n*batch_size+m]
                single_img = cv2.imread(os.path.join(data_path, i[1], i[0]), cv2.IMREAD_COLOR)
                if single_img.shape[2]!=3: # 过滤掉单色图片
                    print("----> ", os.path.join(data_path, i[1], i[0]))
                    continue
                rppg_s = get_rppg_pred(single_img)
                rppg_s = rppg_s.T

                single_img = cv2.resize(single_img, dim)
                single_x = img_to_array(single_img)
                single_x = np.expand_dims(single_x, axis=0)

                if X1 is None:
                    X1 = single_x
                    X2 = rppg_s
                else:
                    X1 = np.append(X1, single_x, axis=0)
                    X2 = np.append(X2, rppg_s, axis=0)

                if i[1]=='1':
                    yv = np.array([[1.0, 0.0]])
                else:
                    yv = np.array([[0.0, 1.0]])

                if y is None:
                    y = yv
                else:
                    y = np.append(y, yv, axis=0)
 
            #return X1, X2, y
            yield [X1, X2], y


train_generator = data_generator(train_dir, batch_size)
val_generator = data_generator(val_dir, 1)

# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
#model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")
#print("[INFO] Model is loaded from disk")

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr = learning_rate), loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# train the model on the new data for a few epochs
model.fit(train_generator, 
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=3000
)

model.save('batch_%d_epochs_%d_steps_%d_0.h5'%(batch_size, epochs, steps_per_epoch))

