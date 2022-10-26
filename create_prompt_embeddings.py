import os, sys 
import numpy as np 

import torch

import clip 

from videoloader import CLIPVideoDataset

from utils.cifar100_classes import classes as cif_classes 
from utils.cifar100_prompts import templates as cif_templates 

from utils.imagenet_classes import imagenet_classes
from utils.imagenet_templates import imagenet_templates

from utils.kinectics700_classes import classes as kin_classes 
from utils.kinectics700_prompts import templates as kin_templates 

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_prompt_embeddings(classes, prompts, model):  

    all_prompts = []
    for ctext in classes:
        for prompt in prompts:
            p = prompt.format(ctext)
            all_prompts.append(p)
    
    prompt_tokens = clip.tokenize(all_prompts).to(device)
    
    # print('prompt_tokes:', len(prompt_tokens))

    batch_size = 2000
    num_batches = len(prompt_tokens) // batch_size

    prompt_features = None 

    with torch.no_grad():
        for i in range(num_batches):
            s = i*batch_size 
            e = (i+1)*batch_size 
            batch_features = model.encode_text(prompt_tokens[s:e])

            if isinstance(prompt_features, type(None)):
                prompt_features = torch.clone(batch_features)
            else:
                prompt_features = torch.cat((prompt_features, batch_features), dim=0)
            print('prompt_features:', prompt_features.shape)
        
        if num_batches*batch_size < len(prompt_tokens) - 1:
            e = num_batches*batch_size
            batch_features = model.encode_text(prompt_tokens[e:])
            if isinstance(prompt_features, type(None)):
                prompt_features = torch.clone(batch_features)
            else:
                prompt_features = torch.cat((prompt_features, batch_features), dim=0)

    print('prompt_features:', prompt_features.shape)
    return prompt_features

def create_and_save_cifar_prompt_embeddings():
    model, preprocess = clip.load("ViT-B/32", device=device)

    #cifar 
    print('-'*10, 'CIFAR100', '-'*10)
    num_classes = len(cif_classes)
    num_prompts = len(cif_templates)

    print('cifar: classes:', num_classes, 'num_prompts:', num_prompts, 'c.p', num_classes * num_prompts)
    prompt_features = create_prompt_embeddings(cif_classes, cif_templates, model)
    print('prompt_features:',prompt_features.shape)

    save = {
        'cifar' : prompt_features
    }
    torch.save(save, 'data/prompt_embeddings/cifar_class_prompt_embeds.pt')
    
    del model
    torch.cuda.empty_cache()

def create_and_save_imagenet_prompt_embeddings():
    model, preprocess = clip.load("ViT-B/32", device=device)

    #kinectics
    print('-'*10, 'ImageNet', '-'*10)
    num_classes = len(imagenet_classes)
    num_prompts = len(imagenet_templates)

    print('imagenet: classes:', num_classes, 'num_prompts:', num_prompts, 'c.p', num_classes * num_prompts)
    prompt_features = create_prompt_embeddings(imagenet_classes, imagenet_templates, model)
    print('prompt_features:',prompt_features.shape)

    save = {
        'imagenet' : prompt_features
    }
    torch.save(save, 'data/prompt_embeddings/imagenet_class_prompt_embeds.pt')
    
    del model
    torch.cuda.empty_cache()

def create_and_save_kinetics_prompt_embeddings():
    model, preprocess = clip.load("ViT-B/32", device=device)

    #kinectics
    print('-'*10, 'Kinetics700', '-'*10)
    num_classes = len(kin_classes)
    num_prompts = len(kin_templates)

    print('kinetics: classes:', num_classes, 'num_prompts:', num_prompts, 'c.p', num_classes * num_prompts)
    prompt_features = create_prompt_embeddings(kin_classes, kin_templates, model)
    print('prompt_features:',prompt_features.shape)

    save = {
        'kinectics' : prompt_features
    }
    torch.save(save, 'data/prompt_embeddings/kinectics_class_prompt_embeds.pt')
    
    del model
    torch.cuda.empty_cache()