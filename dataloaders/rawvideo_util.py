import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
# pip install opencv-python
import cv2
cv2.setNumThreads(0)

# Based on https://github.com/ArrowLuo/CLIP4Clip
class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=1, type=0, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.type = type
        self.transform = self._transform(self.size)
        

    def _transform(self, n_px):
        if self.type == 0:
            return Compose([
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        #objects, iou, temporal, action
        if self.type == 1:
            return Compose([transforms.ToTensor()])
        # i3d
        if self.type == 2:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            
            return Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
       

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):

        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data},video_data.shape

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input,shapes = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input,shapes

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor


# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2