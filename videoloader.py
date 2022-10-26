import os, sys
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader

from decord import VideoReader
from decord import cpu 
from PIL import Image 
import clip 

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPVideoDataset(Dataset):
    def __init__(self, video_path, clip_preprocess):
        super(CLIPVideoDataset, self).__init__()
        self.video_path = video_path

        with open(video_path, 'rb') as f:
            self.decord_video = VideoReader(f, ctx=cpu(0))
        self.video_fps = self.decord_video.get_avg_fps()

        self.total_frames = len(self.decord_video)
        self.sampled_frame_indices = np.arange(0, self.total_frames, self.video_fps)
        self.sampled_frames = self.decord_video.get_batch(self.sampled_frame_indices).asnumpy()

        # print('ssampled_frams:', self.sampled_frames.shape)
        
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.sampled_frame_indices)

    def __getitem__(self, idx):
        frame = np.copy(self.sampled_frames[idx])
        pil_image = Image.fromarray(frame.astype('uint8'), 'RGB')
        torch_image = self.clip_preprocess(pil_image)
        # print('torch image:', torch_image.shape)

        return torch_image

# model, preprocess = clip.load("ViT-B/32", device=device)

# vr = CLIPVideoDataset('data/v/0AIb1TdRh_M.mp4', preprocess)
# vr_loader = DataLoader(vr, 128, shuffle=False, num_workers=0)

# clip_features = None
# for idx, batch in enumerate(vr_loader):
#     print(idx, batch.shape)
#     with torch.no_grad():
#         batch_features = model.encode_image(batch.to(device))
#     if isinstance(clip_features, type(None)):
#         clip_features = batch_features
#     else:
#         clip_features = torch.cat((clip_features, batch_features), dim=0)
#     # sys.exit()