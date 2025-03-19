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

# 1. Подготовка данных

## Как прочитать словарь, переданный вами внутри архива - используйте эту функцию в своём датасете
def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    ...
    return tok_to_ind, ind_to_tok

## Ваш датасет
class ImageCaptioningDataset(Dataset):
    """
        imgs_path ~ путь к папке с изображениями
        captions_path ~ путь к .tsv файлу с заголовками изображений
    """
    def __init__(self, imgs_path, captions_path, train=False):
        super(ImageCaptioningDataset).__init__()
        ...

    def __getitem__(self, index):
        ...
        return img, captions
    
    def __len__(self):
        return ...

## Ваш даталоадер
def collate_fn(batch):
    ...

def get_val_dataloader(dataset, batch_size):
    ...

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
