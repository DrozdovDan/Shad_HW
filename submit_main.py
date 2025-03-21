# Описание структуры присылаемого архива - для самопроверки
#  - submit_main.py
#  - vocab.tsv
#  - checkpoint

# ВАЖНО: если в любой функции есть параметры - не меняйте их порядок и не переименовывайте,
#   если требуется добавить ещё параметры, то добавляйте в конец и обязательно с установленными default-ами

# 0. Все необходимые import-ы
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import cv2
from torchvision import transforms as tr
from torchvision import models
import numpy as np
import re
import string

def tokenize(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]+\ *", " ", text)
    text = text.strip()
    text = text.split()
    text = ['<BOS>'] + text + ['<EOS>']
    return text

# 1. Подготовка данных

## Как прочитать словарь, переданный вами внутри архива - используйте эту функцию в своём датасете
def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    vocab_path = os.path.join(unzip_root, "vocab.tsv")
    vocab = pd.read_csv(vocab_path, sep='\t')
    tok_to_ind = {
        '<UNK>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
        '<PAD>': 3,
    }
    ind_to_tok = {
        0: '<UNK>',
        1: '<BOS>',
        2: '<EOS>',
        3: '<PAD>',
    }
    i = max(ind_to_tok.keys()) + 1
    for _, row in vocab.iterrows():
        if row['token'] not in tok_to_ind.keys():
            tok_to_ind[row['token']] = i
            ind_to_tok[i] = row['token']
            i += 1
    return tok_to_ind, ind_to_tok

def to_ids(text, tok_to_ind):
    return list(map(lambda x: tok_to_ind[x] if x in tok_to_ind else tok_to_ind['<UNK>'], tokenize(text)))

weights = models.ResNet18_Weights.DEFAULT
channel_mean = np.array(weights.transforms().mean)
channel_std = np.array(weights.transforms().std)

image_prepare = tr.Compose([
    tr.ToPILImage(),
    # Любые преобразования, которые вы захотите:
    #   https://pytorch.org/vision/stable/transforms.html
    tr.Resize([256, 256], tr.InterpolationMode.BILINEAR),
    tr.RandomCrop([224, 224]),
    tr.ToTensor(),
    tr.Normalize(mean=channel_mean, std=channel_std),
])

image_prepare_val = tr.Compose([
    tr.ToPILImage(),
    tr.Resize([224, 224], tr.InterpolationMode.BILINEAR),
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
        # Читаем и записываем из файлов в память класса, чтобы быстро обращаться внутри датасета
        # Если не хватает памяти на хранение всех изображений, то подгружайте прямо во время __getitem__, но это замедлит обучение
        # Проведите всю предобработку, которую можно провести без потери вариативности датасета, здесь
        self.train = train
        self.imgs_path = imgs_path
        self.captions_path = captions_path

        self.items = []

        tok_to_ind, ind_to_tok = get_vocab("./")

        df = pd.read_csv(captions_path, sep='\t')
        for i, row in df.iterrows():
          image = cv2.imread(os.path.join(imgs_path, row['img_id']))

          captions = [to_ids(row[label], tok_to_ind) for label in df.keys() if label != 'img_id']
          self.items.append({'image': image, 'captions': captions})

    def __getitem__(self, index):
        item = self.items[index]

        if self.train:
          img = image_prepare(item['image'])
        else:
          img = image_prepare_val(item['image'])

        # Получаем предобработанное изображение (не забудьте отличие при train=True или train=False)
        captions = item['captions']

        # Берём все заголовки или только один случайный (случайность должна происходить при каждом вызове __getitem__,
        #  чтобы во время обучения вы в разных эпохах могли видеть разные заголовки для одного изображения)

        return img, captions
    
    def __len__(self):
        return len(self.items)

## Ваш даталоадер
def collate_fn(batch):
    # Функция получает на вход batch - представляет из себя List[el], где каждый el - один вызов __getitem__
    #  вашего датасета
    # На выход вы выдаёте то, что будет выдавать Dataloader на каждом next() из генератора - вы хотите иметь на выходе
    #  несколько тензоров

    # Моё предложение по тому как должен выглядеть батч на выходе:
    #   img_batch: [batch_size, num_channels, height, width] --> сложенные в батч изображения
    #   captions_batch: [batch_size, num_captions_per_image, max_seq_len or local_max_seq_len] --> сложенные в
    #       батч заголовки при помощи padding-а
    tok_to_ind, ind_to_tok = get_vocab("./")
    global_max_seq_len = np.max(list(map(lambda x: np.max([len(y) for y in x[1]]), batch)))
    img_batch = torch.stack([img for img, cap in batch], dim=0)
    captions_batch = torch.stack([torch.stack([nn.functional.pad(torch.Tensor(cap).to(torch.int32), (0, global_max_seq_len - len(cap) + 2), value=tok_to_ind['<PAD>']) for cap in caps], dim=0) for img, caps in batch], dim=0)
    return img_batch, captions_batch

def get_val_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

# 2. Построение модели

class img_fe_class(nn.Module):
    def __init__(self: int, 
                 pretrained_model='resnet18',
                 freeze_layers='all',
                 unfreeze_last: int = 0):
        super(img_fe_class, self).__init__()
        if pretrained_model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
            base_features_dim = 512
        elif pretrained_model == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            base_model = models.resnet34(weights=weights)
            base_features_dim = 512
        elif pretrained_model == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
            base_features_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {pretrained_model}")

        self.img_feature_dim = base_features_dim
        modules = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        if unfreeze_last > 0:
            children = list(self.backbone.children())
            total = len(children)
            for i, child in enumerate(children):
                if i < total - unfreeze_last:
                    for param in child.parameters():
                        param.requires_grad = False
        else:
            if freeze_layers == 'all':
                for param in self.backbone.parameters():
                    param.requires_grad = False
            elif freeze_layers == 'none':
                pass

    def forward(self, imgs):
        
        features = self.backbone(imgs)
        features = features.reshape(features.size(0), -1)
        return features

from einops import rearrange

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
        self.rnn_type = rnn_type.lower()
        
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embed_dim, padding_idx=tok_to_ind['<PAD>'])
        '''self.embed.weight = nn.Parameter(
            torch.from_numpy(glove_weights).to(dtype=self.embed.weight.dtype),
            requires_grad=False,
        )'''

        if img_feature_dim != hidden_dim:
            self.img_to_hidden = nn.Linear(img_feature_dim, hidden_dim)
        else:
            self.img_to_hidden = nn.Identity()

        # Определение типа RNN
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

        # Преобразование фичей изображений в нужную размерность
        img_features = self.img_to_hidden(img_features)  # [batch_size, hidden_dim]
        
        # Преобразование текстов для пакетной обработки
        texts_flat = rearrange(texts, "bs cap seq -> (bs cap) seq")  # [batch_size * num_captions, seq_len]
        
        # Эмбеддинг текстовых токенов
        embedded = self.embed(texts_flat)  # [batch_size * num_captions, seq_len, embed_dim]
        
        # Подготовка фичей изображений для использования в качестве начального скрытого состояния
        # Сначала размножаем для каждого описания (caption)
        h_0 = img_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        h_0 = h_0.repeat(1, num_captions, 1)  # [batch_size, num_captions, hidden_dim]
        h_0 = rearrange(h_0, "bs cap hidden -> (bs cap) hidden")  # [batch_size * num_captions, hidden_dim]
        
        # Затем преобразуем для слоев и направлений RNN
        h_0 = h_0.unsqueeze(0)  # [1, batch_size * num_captions, hidden_dim]
        h_0 = h_0.repeat(self.num_layers, 1, 1)  # [num_layers * direction_factor, batch_size * num_captions, hidden_dim]
        
        # Обработка в зависимости от типа RNN
        if self.rnn_type == 'lstm':
            # Для LSTM нужно также состояние ячейки (c_0)
            c_0 = torch.zeros_like(h_0)
            outputs, _ = self.rnn(embedded, (h_0, c_0))
        else:
            # Для RNN/GRU нужно только скрытое состояние
            outputs, _ = self.rnn(embedded, h_0)

        # Преобразуем выходы обратно, чтобы включить размерность описаний
        outputs = rearrange(outputs, "(bs cap) seq hidden -> bs cap seq hidden", 
                           bs=batch_size, cap=num_captions)
        
        # Возвращаем только outputs, как ожидает тестовая ячейка
        return outputs
    
from collections import OrderedDict

class image_captioning_model(nn.Module):
    def __init__(self, 
                 vocab_size, 
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
        return self.fc(text_features)

## Аргументы для общего класса
init_kwargs = dict('vocab_size' : len(get_vocab('./')[0].keys()))

# 3. Обучение модели

## Сборка вашей модели с нужными параметрами и подгрукой весов из чекпоинта
def get_model(unzip_root: str):
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    ...

# 4. Оценка результатов

## Генерация предсказания по картинке
def generate(
    model,
    image,
    max_seq_len: Optional[int],
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """
    Args:
        model (nn.Module): Модель из функции get_model
    """
    ...
    return result_tokens, result_text
