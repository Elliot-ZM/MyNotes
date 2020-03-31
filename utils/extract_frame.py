# -*- coding: utf-8 -*-


import os,sys
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw,ImageFont
from utils import draw_face

path = '/home/zmh/hdd/Projects/datasets/face_tiktok'

file = [os.path.join(path , i ) for i in os.listdir(path) if i.endswith('.mp4')]

#%%
def display_img(img, video= True):
    cv2.imshow('display', img)
    if video == True:
        if cv2.waitKey(1)==27: 
            cap.release()
            cv2.destroyAllWindows()
            
    else:
        if cv2.waitKey(0)==27:

            cv2.destroyAllWindows()
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# min_face_size
mtcnn = MTCNN(
    image_size=160, 
    thresholds=[0.9, 0.9, 0.95], min_face_size= 20,
    device=device
)
#%%
cap = cv2.VideoCapture(file[1])
basename = os.path.basename(file[1]).split('.mp4')[0]
new_folder = os.path.join(path , basename)
try:
    
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
except OSError:
    print('Error: Creating directory')
    
current_frame =0

while (cap.isOpened()):
    
    ret, frame = cap.read()
    
    if not ret:
        cv2.destroyAllWindows()
        break
    
    frame_name = os.path.join(new_folder, f'img-{current_frame}.jpg' )
    
    print('Creating....'+ frame_name)
    
    pil_img = Image.fromarray(frame[:,:,::-1]) # change PIL image from BGR np-array
    if current_frame%1 == 0:
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
        
        if type(boxes) != type(None):  # check None 'type' for detected boxes
            #draw detected boxes, probs and landmarks 
            result = draw_face(pil_img, boxes,probs, landmarks, file_path=frame_name) 
        else: 
        
            result = frame.copy()
        # cv2.imwrite(frame_name, result)
    else: 
        result = frame.copy()
    
    current_frame +=1
    
    display_img(result, video = True)
    
#%%


def show_landmarks(img, idx):
    cpy = img.copy()
    draw = ImageDraw.Draw(cpy)

    loc = landmarks[0][idx]
    draw.ellipse([loc[0]-2.0, loc[1]-2.0,loc[0]+2.0,loc[1]+2.0], outline=(0,255,0), fill=(0,255,0))
    cpy.show()
    

# show_landmarks(pil_img, -1)

display_img(result,0)
#%%
desiredLeftEye = (0.35,0.35)
desiredFaceWidth = 256
desiredFaceHeight = 256

l_center = landmarks[0][0].astype('int')
r_center =  landmarks[0][1].astype('int')


dY = r_center[1] - l_center[1]
dX = r_center[0] - l_center[0]

angle = np.degrees(np.arctan2(dY,dX)) -180

#%
desiredRightEyeX = 1.0 - desiredLeftEye[0]
dist = np.sqrt((dX ** 2) + (dY ** 2))

desiredDist = (desiredRightEyeX- desiredLeftEye[0])
desiredDist *= desiredFaceWidth
scale = desiredDist / dist

#%

eyesCenter = ((l_center[0] + r_center[0]) // 2,
            (l_center[1] + r_center[1]) // 2)

M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)


tX = desiredFaceWidth * 0.5
tY = desiredFaceHeight * desiredLeftEye[1]
M[0, 2] += (tX - eyesCenter[0])
M[1, 2] += (tY - eyesCenter[1])
(w, h) = (desiredFaceWidth, desiredFaceHeight)
output = cv2.warpAffine(result, M, (w, h),
    flags=cv2.INTER_CUBIC)
































