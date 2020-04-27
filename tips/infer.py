import cv2
from model import BiSeNet
import os
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
from skimage.measure import regionprops, label
import time
from threading import Thread


class BISENET:
    def __init__(self, model_path, csv_path):
        # retrieve label info
        self.label_info = get_label_info(csv_path)

        # build model and load weight
        self.model = BiSeNet(12, 'resnet18')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # retrieve person color
        self.person_color = self.label_info['Pedestrian'][:-1]

    def predict_on_image(self, image):
        # transform image to tensor
        image = self.transform(image[:, :, ::-1]).to(self.device)

        # prediction map
        predict = self.model(image.unsqueeze(0)).squeeze()

        # encode to class index
        predict = reverse_one_hot(predict).cpu().numpy()

        # encode to color code
        predict = colour_code_segmentation(predict, self.label_info).astype(np.uint8)

        # get bbox output
        predict, bboxes, num_people = self.bbox_output(predict)
        return predict, bboxes, num_people

    def bbox_output(self, predict):
        # get a binary mask with persons color white and background black
        person_mask = np.zeros(predict.shape, dtype=np.uint8)
        person_mask[np.all(predict == self.person_color, axis=-1)] = [255, 255, 255]
        person_mask = cv2.cvtColor(person_mask, cv2.COLOR_BGR2GRAY)

        # label the mask image with connected-components algorithm
        label_image = label(person_mask)

        # find the bbox regions
        regions = regionprops(label_image)

        bboxes = []
        num_people = [0]
        i = 1

        for props in regions:
            if props.area > 100:
                minr, minc, maxr, maxc = props.bbox
                bboxes += [props.bbox]
                num_people += [i]
                predict = cv2.rectangle(predict, (minc, minr), (maxc, maxr), (0, 255, 0), 2, cv2.LINE_AA)
                predict = cv2.putText(predict, f'person{i}', (minc, minr-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                i +=1 

        return predict, bboxes, num_people[-1]

class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = time.time()
        return self

    def stop(self):
        self._end = time.time()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return self._end - self._start

    def fps(self):
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    """From PyImageSearch
    Webcam reading with multi-threading
    """
    def __init__(self, src=0, name='WebcamVideoStream'):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.name = name
        self.stopped = False 

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    mode = 'video'  # image or video or webcam
    bisenet = BISENET('cfg/best_dice_loss_mIoU0_67.pth', 'cfg/class_dict.csv')
    
    if mode == 'webcam':
        vs = WebcamVideoStream().start()
        fps = FPS().start()

        while True:
            frame = vs.read()
            vis, out_bboxes, out_person = bisenet.predict_on_image(frame)
            #print(out_bboxes)
            cv2.imshow('frame', vis)
            fps.update()
            if cv2.waitKey(1) == ord('q'):
                break
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        vs.stop()
    
    elif mode == 'image':
        img_file = 'test.png'
        image = cv2.imread(img_file)
        vis, out_bboxes, out_person = bisenet.predict_on_image(image)
        cv2.imwrite('predicted.png', cv2.cvtColor(np.uint8(vis), cv2.COLOR_RGB2BGR))
    
    elif mode == 'video':
        video_file = 'Paris.m4v'
        output_path = video_file.split('.')[0] + '_out.mp4'
        vid = cv2.VideoCapture(video_file)
        if not vid.isOpened():
            raise IOError("Couldn't open video")
        video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != '' else False
        if isOutput:
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        while True:
            val, frame = vid.read()
            if val:
                vis, out_bboxes, out_person = bisenet.predict_on_image(frame)
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.imshow('result', vis)
                if isOutput:
                    out.write(vis)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        vid.release()
        if isOutput:
            out.release()
        cv2.destroyAllWindows()
        
    