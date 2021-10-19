import numpy as np
import cv2
from .util.opencv_util import *
from .rPPG_preprocessing import *
import math

class rPPG_Extracter():
    def __init__(self):
        self.prev_face = [0,0,0,0]
        self.skin_prev = []
        self.rPPG = []
        self.frame_cropped = []
        self.sub_roi_rect = []
        
    def calc_ppg(self,num_pixels,frame):
        #print("--->", num_pixels, frame.shape)
        r_avg = np.sum(frame[:,:,0])/num_pixels
        g_avg = np.sum(frame[:,:,1])/num_pixels
        b_avg = np.sum(frame[:,:,2])/num_pixels
        ppg = [r_avg,g_avg,b_avg]
        for i,col in enumerate(ppg):
            if math.isnan(col):
                ppg[i] = 0
        return ppg

    def process_frame(self, frame, sub_roi, use_classifier):
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frame_cropped,gray_frame,self.prev_face = crop_to_face(frame,gray_frame,self.prev_face)
        frame_cropped,gray_frame = frame,gray_frame

        if len(sub_roi) > 0:
            sub_roi_rect = get_subroi_rect(frame_cropped,sub_roi)
            frame_cropped = crop_frame(frame_cropped,sub_roi_rect)
            gray_frame = crop_frame(gray_frame,sub_roi_rect)
            self.sub_roi_rect = sub_roi_rect

        num_pixels = frame.shape[0] * frame.shape[1]
        if use_classifier:
            frame_cropped,num_pixels = apply_skin_classifier(frame_cropped)
        return frame_cropped, gray_frame,num_pixels

    def measure_rPPG(self,frame,use_classifier = False,sub_roi = []): 
        frame_cropped, gray_frame,num_pixels = self.process_frame(frame, sub_roi, use_classifier)
        self.rPPG.append(self.calc_ppg(num_pixels,frame_cropped))
        self.frame_cropped = frame_cropped


