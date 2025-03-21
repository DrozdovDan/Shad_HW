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
    tr.Resize([256, 256], tr.InterpolationMode.BILINEAR),
    tr.ToTensor(),
    tr.Normalize(mean=channel_mean, std=channel_std),
])

image_prepare_val = tr.Compose([
    tr.ToPILImage(),
    tr.Resize([256, 256], tr.InterpolationMode.BILINEAR),
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

## Аргументы для общего класса
init_kwargs = dict()

## Общий класс модели
class image_captioning_model(nn.Module):
    def __init__(self, ...):
        super(image_captioning_model, self).__init__()
        ...
        
    def forward(self, img_batch, texts_batch):
        ...

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
