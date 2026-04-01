import os
import pandas as pd

from datasets import load_dataset

# функция инициализирует нужный датасет из CSV и возвращает его в pandas
# если файла CSV нет - загружает из сети
def load_sentiment140_csv(data_dir="./data", filename="sentiment140_train.csv"):
    os.makedirs(data_dir, exist_ok=True) # создание директории
    file_path = os.path.join(data_dir, filename) # путь к файлу

    # проверка наличия файла
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="latin-1")
        return df

    # если нет — скачиваем
    print("CSV is not found... Downloading...")
    dataset = load_dataset("sentiment140", trust_remote_code=True)

    # в pandas
    df = dataset['train'].to_pandas()

    # сохраняем в file_path
    df.to_csv(file_path, index=False)
    print(f'Saved to {file_path}')

    return df