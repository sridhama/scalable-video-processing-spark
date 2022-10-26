import os, sys 
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader
import torch. nn as nn 

import clip 

from videoloader import CLIPVideoDataset
from create_prompt_embeddings import create_and_save_imagenet_prompt_embeddings, create_and_save_kinetics_prompt_embeddings, create_and_save_cifar_prompt_embeddings

from utils.cifar100_classes import classes as cifar_classes 
from utils.cifar100_prompts import templates as cifar_templates 
from utils.imagenet_classes import imagenet_classes
from utils.imagenet_templates import imagenet_templates
from utils.kinectics700_classes import classes as kin_classes 
from utils.kinectics700_prompts import templates as kin_templates 

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPTag(nn.Module):
    def __init__(self, model, clip_preprocess):
        super(CLIPTag, self).__init__()

        self.model = model 
        self.clip_preprocess = clip_preprocess

        self.cifar_classes = cifar_classes 
        self.cifar_templates = cifar_templates 

        self.imagenet_classes = imagenet_classes 
        self.imagenet_templates = imagenet_templates 

        self.kin_classes = kin_classes 
        self.kin_templates = kin_templates 

        self.set_prompts_features()

    def set_prompts_features(self):
        # cpath = 'data/prompt_embeddings/cifar_class_prompt_embeds.pt'
        # if not os.path.exists(cpath):
        #     print('creating cifar prompt features')
        #     create_and_save_cifar_prompt_embeddings()
        # else:
        #     print('cifar prompts features found at', cpath)
        # self.cifar_prompt_features = torch.load(cpath)['cifar']

        cpath = 'data/prompt_embeddings/imagenet_class_prompt_embeds.pt'
        if not os.path.exists(cpath):
            print('creating imagenet prompt features')
            create_and_save_imagenet_prompt_embeddings()
        else:
            print('imagenet prompts features found at', cpath)
        self.imagenet_prompt_features = torch.load(cpath)['imagenet']
        # print('cifar_prompts:', self.cifar_prompt_features.device)
        
        kpath = 'data/prompt_embeddings/kinectics_class_prompt_embeds.pt'
        if not os.path.exists(kpath):
            print('creating kinetics prompt features')
            create_and_save_kinetics_prompt_embeddings()
        else:
            print('kinetics prompt features already found at', kpath)
        self.kinetics_prompt_features = torch.load(kpath)['kinectics']
        # print('kinetics_prompts:', self.kinetics_prompt_features.device)

    def extract_video_features(self, video_path):
        vr = CLIPVideoDataset(video_path, self.clip_preprocess)
        vr_loader = DataLoader(vr, 128, shuffle=False, num_workers=0)
        
        clip_features = None 
        for idx, frames_batch in enumerate(vr_loader):
            # print(f'extracting: {idx}/{len(vr_loader)}')
            with torch.no_grad():
                batch_features = self.model.encode_image(frames_batch.to(device))
            if isinstance(clip_features, type(None)):
                clip_features = batch_features
            else:
                clip_features = torch.cat((clip_features, batch_features), dim=0)
            print(f'extracting: {idx}/{len(vr_loader)}', 'clip_features:', clip_features.shape, clip_features.device)

        return clip_features
    
    def tag_video_features(self, video_features, prompt_features, num_classes, num_prompts, topk=5):
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        prompt_features = prompt_features / prompt_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * video_features @ prompt_features.t()

        logits_per_image = logits_per_image.view(-1, num_classes, num_prompts)
        logits_per_image = logits_per_image.mean(dim=-1) # frames x classes

        frame_top_values, frame_top_indices = torch.topk(logits_per_image, topk, dim=-1)

        top_values = frame_top_values.view(-1)
        top_indices = frame_top_indices.view(-1)

        ascending_psued_idx = torch.argsort(top_values)
        ascending_class_idx = torch.gather(top_indices, dim=-1, index=ascending_psued_idx)
        topk_class_idx = ascending_class_idx.cpu().numpy()[::-1][:topk]
        
        unique_top_class_indices = np.unique(topk_class_idx)
        return unique_top_class_indices

    def tag_video_features_cifar(self, video_features, topk=5):
        unique_top_class_indices = self.tag_video_features(video_features, self.cifar_prompt_features, len(self.cifar_classes), len(self.cifar_templates), topk=topk)
        unique_classes = [self.cifar_classes[idx] for idx in unique_top_class_indices]
        return unique_classes

    def tag_video_features_imagenet(self, video_features, topk=5):
        unique_top_class_indices = self.tag_video_features(video_features, self.imagenet_prompt_features, len(self.imagenet_classes), len(self.imagenet_templates), topk=topk)
        unique_classes = [self.imagenet_classes[idx] for idx in unique_top_class_indices]
        return unique_classes
    
    def tag_video_features_kinetics(self, video_features, topk=5):
        unique_top_class_indices = self.tag_video_features(video_features, self.kinetics_prompt_features, len(self.kin_classes), len(self.kin_templates), topk=topk)
        unique_classes = [self.kin_classes[idx] for idx in unique_top_class_indices]
        return unique_classes

    def tag_video(self, video_path, topk=5):
        clip_features = self.extract_video_features(video_path)
        # topk_cifar_classes = self.tag_video_features_cifar(clip_features, topk)
        topk_imagenet_classes = self.tag_video_features_imagenet(clip_features, topk)
        topk_kinetics_classes = self.tag_video_features_kinetics(clip_features, topk)
        return topk_imagenet_classes + topk_kinetics_classes

model, preprocess = clip.load("ViT-B/32", device=device)
clip_tagger = CLIPTag(model, preprocess)

video_path = 'data/v/quantumania.mp4'
tags = clip_tagger.tag_video(video_path)
print(tags)