from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length: int = max_length
        self.texts: list[str] = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> dict[str, list[int]]:
        text = self.texts[idx]

        # токенизация
        tokens = self.tokenizer(
            text, # строка
            truncation=True,  # обрезка до max_length
            max_length = self.max_length,
            add_special_tokens=False
        )["input_ids"]

        # добавление EOS токена
        tokens = tokens + [self.tokenizer.eos_token_id]

        return self._split_inputs_labels(tokens)

    @abstractmethod
    def _split_inputs_labels(self, tokens) -> dict[str, list[int]]:
        ...


class TrainDataset(BaseDataset):
    def _split_inputs_labels(self, tokens) -> dict[str, list[int]]:
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }

class EvalDataset(BaseDataset):
    def _split_inputs_labels(self, tokens) -> dict[str, list[int]]:
        split_idx = int(len(tokens) * 0.75)
        return {
            'input_ids': tokens[:split_idx],
            'labels': tokens[split_idx:]
        }