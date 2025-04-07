# Описание структуры присылаемого архива - для самопроверки
#  - submit_main.py
#  - vocab.tsv
#  - checkpoint

# ВАЖНО: если в любой функции есть параметры - не меняйте их порядок и не переименовывайте,
#   если требуется добавить ещё параметры, то добавляйте в конец и обязательно с установленными default-ами

# 0. Все необходимые import-ы
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
from torchvision.transforms import InterpolationMode

import torch
import cv2
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
# import matplotlib.pyplot as plt

import re

import random

from torchvision import models
from torch import nn

from einops import rearrange

from collections import OrderedDict
import torch.nn.functional as F

from torch.optim.lr_scheduler import MultiStepLR

# 1. Подготовка данных

## Как прочитать словарь, переданный вами внутри архива - используйте эту функцию в своём датасете
def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    vocab_path = os.path.join(unzip_root, "vocab.tsv")
    vocab = pd.read_csv(vocab_path, sep='\t')
    
    ind_to_tok = vocab['0'].to_dict()
    tok_to_ind = {tok: ind for ind, tok in ind_to_tok.items()}
    
    return tok_to_ind, ind_to_tok

tok_to_ind, ind_to_tok = get_vocab('')
vocab_size = len(tok_to_ind)

channel_mean = np.array([0.485, 0.456, 0.406])
channel_std = np.array([0.229, 0.224, 0.225])

image_prepare = tr.Compose([
    tr.ToPILImage(),
    tr.Resize(256, interpolation=InterpolationMode.BILINEAR),
    tr.RandomCrop(224),
    tr.RandomHorizontalFlip(p=0.5),
    tr.RandomVerticalFlip(p=0.1),
    tr.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    tr.RandomGrayscale(p=0.05),
    tr.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    tr.ToTensor(),
    tr.Normalize(mean=channel_mean, std=channel_std),
])


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.strip()
    tokens = re.split(r'\s+', text)
    tokens = ['<BOS>'] + tokens + ['<EOS>']
    return tokens

def to_ids(text):
    tokens = tokenize(text)
    tok_to_ind, ind_to_tok = get_vocab("")
    ids = [tok_to_ind.get(token, tok_to_ind['<UNK>']) for token in tokens]
    return ids

# Для валидации рекомендую использовать минимальное количество аугументаций, чтобы
#  замеряться честно - изменения размера, нормализация (все случайные аугументации
#  не делайте для валидации и обязательно для обучения делайте со средним в нуле)
image_prepare_val = tr.Compose([
    tr.ToPILImage(),
    tr.Resize(256, interpolation=InterpolationMode.BILINEAR),
    tr.CenterCrop(224),
    tr.ToTensor(),
    tr.Normalize(mean=channel_mean, std=channel_std),
])

## Ваш датасет
class ImageCaptioningDataset(Dataset):
    """
        imgs_path ~ путь к папке с изображениями
        captions_path ~ путь к .tsv файлу с заголовками изображений
    """
    def __init__(self, imgs_path, captions_path, train=False):
        super(ImageCaptioningDataset).__init__()
        self.imgs_path = imgs_path
        self.train = train
        
        self.captions_df = pd.read_csv(captions_path, sep='\t')
        
        self.caption_cols = [col for col in self.captions_df.columns if col.startswith('caption #')]
        
        self.transform = image_prepare if train else image_prepare_val

    def __getitem__(self, index):
        img_info = self.captions_df.iloc[index]
        img_id = img_info['img_id']

        img_path = os.path.join(self.imgs_path, img_id)
        img = cv2.imread(img_path)
        
        img = self.transform(img)
        
        # Берём все заголовки или только один случайный (случайность должна происходить при каждом вызове __getitem__, 
        #  чтобы во время обучения вы в разных эпохах могли видеть разные заголовки для одного изображения)
        if self.train:
            caption_col = random.choice(self.caption_cols)
            caption_text = img_info[caption_col]
            caption_ids = to_ids(caption_text)
            return img, caption_ids
        
        all_captions = []
        for col in self.caption_cols:
            caption_text = img_info[col]
            caption_ids = to_ids(caption_text)
            all_captions.append(caption_ids)
        return img, all_captions
    
    def __len__(self):
        return len(self.captions_df)

## Ваш даталоадер
def collate_fn(batch):
    images, captions = zip(*batch)
    img_batch = torch.stack(images, dim=0)
    
    tok_to_ind, ind_to_tok = get_vocab("")

    if isinstance(captions[0], list) and isinstance(captions[0][0], int):
        # train        
        max_length = max(len(caption) for caption in captions)
        
        padded_captions = []
        for caption in captions:
            padded = caption + [tok_to_ind['<PAD>']] * (max_length - len(caption))
            padded_captions.append([padded])
        
        captions_batch = torch.tensor(padded_captions, dtype=torch.long)
    else:
        # val
        max_length = max(len(caption) for caption_list in captions for caption in caption_list)
        
        padded_captions = []
        for caption_list in captions:
            padded_list = []
            for caption in caption_list:
                padded = caption + [tok_to_ind['<PAD>']] * (max_length - len(caption))
                padded_list.append(padded)
            padded_captions.append(padded_list)
        
        captions_batch = torch.tensor(padded_captions, dtype=torch.long)
    
    
    return img_batch, captions_batch

def get_val_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

# 2. Построение модели

## Аргументы для общего класса
init_kwargs = dict()

class img_fe_class(nn.Module):
    def __init__(self, 
                 pretrained_model='resnet18',
                 freeze_layers='all',
                 unfreeze_last: int = 0):
        super(img_fe_class, self).__init__()
#         if pretrained_model == 'resnet18':
#         weights = models.ResNet18_Weights.DEFAULT
#         base_model = models.resnet18(weights=weights)
        base_model = models.resnet18()
        base_features_dim = 512
#         elif pretrained_model == 'resnet34':
#             weights = models.ResNet34_Weights.DEFAULT
#             base_model = models.resnet34(weights=weights)
#             base_features_dim = 512
#         elif pretrained_model == 'resnet50':
#             weights = models.ResNet50_Weights.DEFAULT
#             base_model = models.resnet50(weights=weights)
#             base_features_dim = 2048
#         else:
#             raise ValueError(f"Unsupported model: {pretrained_model}")

        self.img_feature_dim = base_features_dim
        modules = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
#         if unfreeze_last > 0:
#             children = list(self.backbone.children())
#             total = len(children)
#             for i, child in enumerate(children):
#                 if i < total - unfreeze_last:
#                     for param in child.parameters():
#                         param.requires_grad = False
#         else:
#             if freeze_layers == 'all':
#                 for param in self.backbone.parameters():
#                     param.requires_grad = False
#             elif freeze_layers == 'none':
#                 pass

    def forward(self, imgs):
        features = self.backbone(imgs)
        features = features.reshape(features.size(0), -1)
        return features

    
    
class text_fe_class(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 hidden_dim=512, 
                 img_feature_dim=512,
                 num_layers=1, 
                 dropout=0.1,
                 rnn_type='rnn'):
        super(text_fe_class, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = 300
        self.hidden_dim = hidden_dim
        self.img_feature_dim = img_feature_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        tok_to_ind, ind_to_tok = get_vocab('')
        
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embed_dim, padding_idx=tok_to_ind['<PAD>'])
#         self.embed.weight = nn.Parameter(
#             torch.from_numpy(glove_weights).to(dtype=self.embed.weight.dtype),
#             requires_grad=False,
#         )

        self.img_to_hidden = nn.Linear(img_feature_dim, hidden_dim)

        if self.rnn_type == 'lstm':
            model = nn.LSTM
        elif self.rnn_type == 'gru':
            model = nn.GRU
        else:
            model = nn.RNN

        self.rnn = model(
                input_size=self.embed_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        
        
    def forward(self, texts, img_features):
        batch_size, num_captions, seq_len = texts.shape
        img_features = self.img_to_hidden(img_features)
        
        texts_flat = rearrange(texts, "bs cap seq -> (bs cap) seq")
        
        embedded = self.embed(texts_flat)
        
        h_0 = img_features.unsqueeze(1)
        h_0 = h_0.repeat(1, num_captions, 1)
        h_0 = rearrange(h_0, "bs cap hidden -> (bs cap) hidden")
        
        h_0 = h_0.unsqueeze(0)
        h_0 = h_0.repeat(self.num_layers, 1, 1)
        
        if self.rnn_type == 'lstm':
            c_0 = torch.zeros_like(h_0)
            outputs, _ = self.rnn(embedded, (h_0, c_0))
        else:
            outputs, _ = self.rnn(embedded, h_0)

        outputs = rearrange(outputs, "(bs cap) seq hidden -> bs cap seq hidden", 
                           bs=batch_size, cap=num_captions)
        
        return outputs


class image_captioning_model(nn.Module):
    def __init__(self, 
                 vocab_size=vocab_size, 
                 pretrained_model='resnet18',
                 freeze_layers_img='all',
                 unfreeze_last=0,
                 hidden_dim=512, 
                 num_layers=1,
                 rnn_type='rnn'):
        super(image_captioning_model, self).__init__()
        self.img_fe = img_fe_class(
            pretrained_model=pretrained_model,
            freeze_layers=freeze_layers_img,
            unfreeze_last=unfreeze_last,
        )
        img_feature_dim = self.img_fe.img_feature_dim

        self.text_fe = text_fe_class(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            img_feature_dim=img_feature_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        
    def forward(self, img_batch, texts_batch):
        img_features = self.img_fe(img_batch)
        text_features = self.text_fe(texts_batch, img_features)
        text_features = self.fc(text_features)
        return text_features

# 3. Обучение модели

def create_model_and_optimizer(model_class, model_params, optimizer, lr, device='cpu'):
    model = model_class(**model_params)
    model = model.to(device)
    
    optimizer = optimizer([p for p in model.parameters() if p.requires_grad], lr=lr)
    return model, optimizer

## Сборка вашей модели с нужными параметрами и подгрукой весов из чекпоинта
def get_model(unzip_root: str):
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    pretrained_model = 'resnet18'
    hidden_dim = 256
    unfreeze_last = 2
    num_layers = 2
    rnn_type = 'gru'

    model_name = f"{pretrained_model}_{hidden_dim}_{unfreeze_last}_{num_layers}_{rnn_type}#0"

    model, optimizer = create_model_and_optimizer(
        model_class=image_captioning_model,
        model_params={
            "vocab_size": vocab_size,
            "pretrained_model": pretrained_model,
            "hidden_dim": hidden_dim,
            "unfreeze_last": unfreeze_last,
            "num_layers": num_layers,
            "rnn_type": rnn_type
        },
        optimizer=torch.optim.Adam,
        lr=1e-3
    )
    scheduler = MultiStepLR(optimizer, milestones=[30, 40, 50], gamma=0.7)
    
#     chkp_path = vocab_path = os.path.join(unzip_root, f"{model_name}.pt")
#     checkpoint = torch.load(chkp_path, weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])

    return model


# 4. Оценка результатов

## Генерация предсказания по картинке
def generate(
    model,
    image,
    max_seq_len: Optional[int],
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    greedy=False
):
    """
    Args:
        model (nn.Module): Модель из функции get_model
    """
    assert top_p is None or top_k is None, "Don't use top_p and top_k at the same time"
    
    model.eval()
    tok_to_ind, ind_to_tok = get_vocab('')

    image = image_prepare_val(image)
    image = image.unsqueeze(0)
    
    generated_tokens = [tok_to_ind['<BOS>']]
    with torch.no_grad():
        for _ in range(max_seq_len):
            input_seq = torch.tensor([[generated_tokens]]).to(device)
            image = image.to(device)
            outputs = model(image, input_seq)
            logits = outputs[0, 0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                next_token = torch.argmax(probs)
            elif top_k is not None:
                topk_probs, topk_indices = torch.topk(probs, top_k)
                topk_probs = topk_probs / topk_probs.sum()
                next_token = topk_indices[torch.multinomial(topk_probs, 1)]
            elif top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
                if len(cutoff) > 0:
                    cutoff_index = cutoff[0] + 1
                else:
                    cutoff_index = len(sorted_probs)
                filtered_probs = sorted_probs[:cutoff_index]
                filtered_indices = sorted_indices[:cutoff_index]
                filtered_probs = filtered_probs / filtered_probs.sum()
                next_token = filtered_indices[torch.multinomial(filtered_probs, 1)]
            else:
                next_token = torch.multinomial(probs, 1)
            token_int = next_token.item()
            if token_int == tok_to_ind['<EOS>']:
                generated_tokens.append(token_int)
                break
            generated_tokens.append(token_int)
    result_text = " ".join([ind_to_tok[token] for token in generated_tokens])
    return generated_tokens, result_text
