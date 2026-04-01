import os
import re

import torch
from torch import nn


def get_device():
    if torch.backends.mps.is_available():
       device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


# функция для "чистки" текстов
def clean_string(text):
    # приведение к нижнему регистру
    text = text.lower()
    # ссылки
    text = re.sub(r'http\S+|www\S+', '', text)

    # упоминания
    text = re.sub(r'@\w+', '', text)

    # хэштеги (оставим слово, уберём #)
    text = re.sub(r'#(\w+)', r'\1', text)

    # эмодзи и спецсимволы
    text = re.sub(r'[^\w\s]', '', text)

    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# функция для сохранения весов модели в словарь state_dict
def save_model_dict(
        model: nn.Module,
        model_dir='./model',
        file_name='lstm.pth'
):
    os.makedirs(model_dir, exist_ok=True)  # создание директории
    file_path = os.path.join(model_dir, file_name)  # путь к файлу
    torch.save(model.state_dict(), file_path)

# функция для загрузки весов модели из словаря state_dict
def load_model_params_from_dict(
        model: nn.Module,
        model_dir='./model',
        file_name='lstm.pth',

) -> nn.Module | None:
    file_path = os.path.join(model_dir, file_name) # путь к файлу
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        model.to(get_device())
        model.eval()  # для инференса
        return model
    else:
        print('state_dict file is not found... :(')
        return None
