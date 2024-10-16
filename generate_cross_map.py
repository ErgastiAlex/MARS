model_path_base = ''
config_path_base = ''

config_path_ours = ''
model_path_ours = ''
use_cuda = True

test_file = '/home/datasets/CUHK-PEDES/CUHK-PEDES/processed_data/test.json'
test_image_root = '/home/datasets/CUHK-PEDES/CUHK-PEDES/imgs'

from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertForMaskedLM
from models.tokenization_bert import BertTokenizer

import torch
from torch import nn
from torchvision import transforms

import json
import spacy

class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(self.text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.mlm_probability = config['mlm_probability']
        self.mrtd_mask_probability = config['mrtd_mask_probability']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(self.text_width, 2)
        self.prd_head = nn.Linear(self.text_width, 2)
        self.mrtd_head = nn.Linear(self.text_width, 2)
        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(self.text_width, embed_dim)
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        # self.copy_params()
        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        
    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids, 
                                attention_mask = text.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                               )     
           
        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings)   
        return vl_output

import re

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])            
    return caption

from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

import ruamel_yaml as yaml
from models.model_person_search import ALBEF


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = yaml.load(open(config_path_base, 'r'), Loader=yaml.Loader)
model_base = ALBEF(text_encoder='bert-base-uncased', tokenizer = tokenizer, config = config)

checkpoint = torch.load(model_path_base, map_location='cpu')              
msg = model_base.load_state_dict(checkpoint["model"],strict=False)
model_base.eval()

block_num = 7

model_base.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

if use_cuda:
    model_base.cuda() 

import ruamel_yaml as yaml
from models.model_person_search import ALBEF


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = yaml.load(open(config_path_ours, 'r'), Loader=yaml.Loader)
model_ours = ALBEF(text_encoder='bert-base-uncased', tokenizer = tokenizer, config = config)

checkpoint = torch.load(model_path_ours, map_location='cpu')              
msg = model_ours.load_state_dict(checkpoint["model"],strict=False)
model_ours.eval()

block_num = 7

model_ours.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

if use_cuda:
    model_ours.cuda() 

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from dataset.ps_dataset import ps_eval_dataset

image_res = 384

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


    
test_transform = transforms.Compose([
    transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

test_dataset = ps_eval_dataset(test_file, test_transform, test_image_root, 70)
loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

import os


def compute_gradcam(model, text_input, image, block_num=7):
    
    encoder_output = model.visual_encoder(image) 
    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to("cuda")
    output = model.text_encoder.bert(text_input.input_ids,
                                         attention_mask=text_input.attention_mask,
                                         encoder_hidden_states=encoder_output.to("cuda"),
                                         encoder_attention_mask=encoder_att,
                                         return_dict=True,
                                         mode='multi_modal'
                                         )
    vl_embedding = output.last_hidden_state[:,0,:]
    output = model.itm_head(vl_embedding)
    loss = output[:,1].sum()

    model.zero_grad()
    loss.backward()    

    with torch.no_grad():
        mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

        grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
        cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()

        cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
        grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask

        gradcam = cams * grads
        gradcam = gradcam[0].mean(0).cpu().detach().numpy()
    return gradcam

nlp = spacy.load('en_core_web_sm')
def generate_gmap(index):

    image_path = os.path.join(test_dataset.image_root, test_dataset.ann[index]['file_path'])
    image = Image.open(image_path).convert('RGB')

    image = test_transform(image)
    image_unnorm =  image.clone()
    image = normalize(image)

    caption =  test_dataset.ann[index]['captions'][0]

    caption = pre_caption(caption)
    text_input = tokenizer(caption, return_tensors="pt")

    caption = " ".join([tokenizer.decode([token_id]) for token_id in text_input.input_ids[0][1:]])

    if use_cuda:
        image = image.cuda()
        text_input = text_input.to(image.device)

    image.unsqueeze_(0)

    gradcam_base = compute_gradcam(model_base, text_input, image, block_num=block_num)
    gradcam_ours = compute_gradcam(model_ours, text_input, image, block_num=block_num)

    num_image = len(text_input.input_ids[0]) 
    fig, ax = plt.subplots( 2 ,num_image, figsize=(num_image*0.7, 3))
    fig.tight_layout()

    os.makedirs('gradcam-t', exist_ok=True)
    rgb_image = image_unnorm.cpu().numpy().transpose(1,2,0)
    rgb_image = cv2.resize(rgb_image, (128, 384))

    ax[0,0].imshow(rgb_image)
    ax[0,0].set_yticks([])
    ax[0,0].set_xticks([])
    ax[0,0].set_xlabel("Base")
    ax[1,0].imshow(rgb_image)
    ax[1,0].set_yticks([])
    ax[1,0].set_xticks([])
    ax[1,0].set_xlabel("Ours")

    idx_words=0
    char_count=0

    count=1
    doc = nlp(caption)
    text_words = caption.split()
    attribute_mask = torch.zeros(len(text_words))
    for chunk in doc.noun_chunks:
        # adj = []
        # noun = ""
        pos=[]
        adj_founded=False # just to be sure
        for tok in chunk:
            if tok.pos_ == "NOUN":
                # noun = tok.text
                #manage she's to be split in she is
                while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
                    char_count+=len(text_words[idx_words])+1
                    idx_words+=1
                pos.append(idx_words) # +1 to account for [CLS]
            if tok.pos_ == "ADJ":
                adj_founded=True
                while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
                    char_count+=len(text_words[idx_words])+1
                    idx_words+=1
                pos.append(idx_words) # +1 to account for [CLS]
        
        if len(pos)>1 and adj_founded:
            print(chunk)
            attribute_mask[pos]=count
            count+=1


    color =['black', 'red', 'blue', 'green', 'orange', 'pink', 'brown', 'gray',  'yellow', 'purple',]

    for i,(token_id, color_i) in enumerate(zip(text_input.input_ids[0][1:],attribute_mask)):
        word = tokenizer.decode([token_id])
        gradcam_image = getAttMap(rgb_image, gradcam_base[i+1])
        gradcam_image = cv2.resize(gradcam_image, (128, 384))*255
        #convert to int32
        gradcam_image = gradcam_image.astype(np.uint8)

        ax[0,i+1].imshow(gradcam_image)
        ax[0,i+1].set_yticks([])
        ax[0,i+1].set_xticks([])
        ax[0,i+1].set_xlabel(word)
        ax[0,i+1].xaxis.label.set_color(color[int(color_i)])
    


    for i,(token_id, color_i) in enumerate(zip(text_input.input_ids[0][1:],attribute_mask)):
        word = tokenizer.decode([token_id])
        gradcam_image = getAttMap(rgb_image, gradcam_ours[i+1])
        #resize to 128 384
        gradcam_image = cv2.resize(gradcam_image, (128, 384))*255
        #convert to int32
        gradcam_image = gradcam_image.astype(np.uint8)

        ax[1,i+1].imshow(gradcam_image)
        ax[1,i+1].set_yticks([])
        ax[1,i+1].set_xticks([])
        ax[1,i+1].set_xlabel(word)
        ax[1,i+1].xaxis.label.set_color(color[int(color_i)])


    plt.subplots_adjust(wspace=0.01, hspace=0.2)

    fig.savefig(f'gradcam-t/radcam_{index}.png')




#generate 30 random index

# import random
# max_index = len(test_dataset.ann)

for i in range(150):
    # index = random.randint(0,max_index)
    generate_gmap(i)
