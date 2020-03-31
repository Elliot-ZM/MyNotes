# import cv2
from PIL import Image,ImageDraw, ImageFont
import numpy as np
      
def draw_face(img, boxes,probs, landmarks, color=(0,190,0),file_path = None):
    img_copy = img.copy()
    
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.truetype(r'font/arial.ttf', 15)
    
    #draw text and bbox
    for prob,box,landmark in zip(probs, boxes, landmarks):
        # if prob*100 >= 95:    # no need to set this threshold, just set last threshold value at mtcnn object  
        draw.rectangle(box.tolist(), outline=color)
        
        text = '%.2f%%'%(prob*100)
        label_size = draw.textsize(text, font)
                
        if box[1] - label_size[1] >= 0:
            text_origin = np.array([box[0], box[1] - label_size[1]])
        else:
            text_origin = np.array([box[0], box[1] + 1])
            
        draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=color)
                        
        #draw text
        draw.text(text_origin, text, font = font, align ="left")
        #draw landmarks
        for lm in landmark:
            draw.ellipse([
               (lm[0]-2.0, lm[1]-2.0),
               (lm[0]+2.0, lm[1]+2.0)
            ], outline= (0,255,0))

        if file_path is not None:
            face = img.crop(box).copy().resize((160,160), Image.BILINEAR)
            # face = img.crop(box).copy()
            face.save(file_path)
            
    np_img = np.array(img_copy, dtype=np.uint8)
    
    return np_img[:,:,::-1]