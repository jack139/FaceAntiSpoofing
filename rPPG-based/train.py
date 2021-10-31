# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint

#########################
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
#########################
from rPPG2.rPPG_Extracter import *


batch_size = 128
#steps_per_epoch = 250 # 20k
steps_per_epoch = 326 # 30k
epochs = 10
learning_rate=2e-4
data_dir = '/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof_Croped/Data'
#train_csv = 'high_20k_nuaa_train.csv'
train_csv = 'high_30k_nuaa_train.csv'
val_csv = 'high_30k_nuaa_test.csv'


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
def data_generator(data_path, data_csv, batch_size):
    file_list = []
    for i in open(os.path.join(data_path, data_csv)):
        if i.strip()=='':
            continue
        file_list.append(i.strip().split(','))

    random.shuffle(file_list)

    print(data_path, ": ", len(file_list), "\tbatch: ", batch_size)

    while True:
        for n in range(len(file_list)//batch_size):
            X1 = X2 = y = None
            for m in range(batch_size):
                i = file_list[n*batch_size+m]
                single_img = cv2.imread(os.path.join(data_path, i[0]), cv2.IMREAD_COLOR)
                if single_img.shape[2]!=3: # 过滤掉单色图片
                    print("----> ", os.path.join(data_path, i[0]))
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


train_generator = data_generator(data_dir, train_csv, batch_size)
val_generator = data_generator(data_dir, val_csv, batch_size)

# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)

# load weights into new model
model.load_weights("batch_128_epochs_10_steps_326_0.h5")
print("[INFO] Model is loaded from disk")

model.compile(optimizer=Adam(lr = learning_rate), loss='categorical_crossentropy', metrics = ['categorical_accuracy'])
model.summary()

model_checkpoint = ModelCheckpoint('batch_%d_epochs_%d_steps_%d_0.h5'%(batch_size, epochs, steps_per_epoch), 
    monitor='val_categorical_accuracy',verbose=1, save_best_only=True, save_weights_only=True)

model.fit(train_generator, 
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=65, # 8386 / 128
    callbacks=[model_checkpoint]
)

#model.save('batch_%d_epochs_%d_steps_%d_0.h5'%(batch_size, epochs, steps_per_epoch))
