# модуль с дополнительными вспомогательными функциями
import os
import re
from typing import Any

import pandas as pd
import torch
from torch import nn


# возвращает доступное в системе GPU-устройство
def get_device():
    if torch.backends.mps.is_available():
       device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


# функция для "чистки" текстов для LSTM
def clean_string_lstm(text):
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


# чистка текстов для GPT трансформера (менее агрессивная)
def clean_string_transformer(text: str) -> str:
    # убираем ссылки
    text = re.sub(r'http\S+|www\S+', '', text)

    # убираем html-теги (если есть)
    text = re.sub(r'<.*?>', '', text)

    # заменяем переносы строк на пробел
    text = re.sub(r'\n+', ' ', text)

    # убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# функция для фильтрации текстов короче минимальной длины
def drop_short_texts(texts: pd.Series, min_length: int) -> pd.Series:
    mask = texts.str.split().apply(
        lambda x: len(x) >= min_length
    )
    return texts[mask].reset_index(drop=True)


# функция для фильтрации текстов длинее максимальной длины
def drop_long_texts(texts: pd.Series, max_length: int) -> pd.Series:
    mask = texts.str.split().apply(
        lambda x: len(x) <= max_length
    )
    return texts[mask].reset_index(drop=True)


# функция для обрезки EOS токенов
def trim_after_eos(tokens, eos_id):
    if eos_id in tokens:
        return tokens[:tokens.index(eos_id)]
    return tokens


# # функция для обрезки паддингов
def trim_padding(tokens, pad_token_id):
    if pad_token_id in tokens:
        idx = tokens.index(pad_token_id)
        return tokens[:idx]
    return tokens


# функция для разделения токенизированного текста на 3/4 и 1/4
def split_tokens(tokens, ratio=0.75):
    split_idx = int(len(tokens) * ratio)
    return {
        'input_ids': tokens[:split_idx],
        'labels': tokens[split_idx:]
    }


# функция для сдвига токенизированного текста на 1 шаг (формирование обучающих примеров)
def shift_tokens(tokens, step=1):
    return {
        'input_ids': tokens[:-step],
        'labels': tokens[step:]
    }


# функция для разделения текста на 3/4 и 1/4 по количеству слов
def split_text(text, ratio=0.75):
    splitted = text.split()
    split_idx = int(len(splitted) * ratio)
    inputs = ' '.join(splitted[:split_idx])
    references = ' '.join(splitted[split_idx:])
    return inputs, references


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


# функция для подготовки данных для модели Transformer
def prepare_data_for_transformer(
        texts: pd.Series, min_length: int, max_length: int
) -> dict[str, list[Any]]:
    # чистка текстов для трансформера
    texts = texts.apply(clean_string_transformer)

    # удаление коротких текстов
    texts = drop_short_texts(texts, min_length)

    # удаление длинных текстов
    texts = drop_long_texts(texts, max_length)

    # разбивка на input (3/4), reference (1/4)
    texts = texts.apply(split_text)

    return {
        'input_ids': [row[0] for row in texts],
        'references': [row[1] for row in texts]
    }
