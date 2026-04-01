from typing import Callable, Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from lstm_model import SequencePredictionLSTM
from lstm_generate import generate


def train(
        model: SequencePredictionLSTM,
        n_epochs: int,
        tokenizer: PreTrainedTokenizerBase,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: CrossEntropyLoss,
        optimizer: Adam,
        scheduler: ReduceLROnPlateau,
        evaluate_func: Callable[[list, list, Any], dict],
        max_length: int
) -> None:

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        rouge_metrics: list[float] = []
        for batch in tqdm(train_loader, desc=f'Training. Epoch {epoch}'):
            inputs = batch['input_ids']
            labels = batch['labels']
            lengths = batch['lengths']
            optimizer.zero_grad()
            logits = model.forward(inputs, lengths)
            loss = criterion(
                logits.view(-1, len(tokenizer.get_vocab())),
                labels.view(-1)
            )
            # считаем градиенты
            loss.backward()
            # обновляем веса
            optimizer.step()
            total_loss += loss.item()
            scheduler.step(loss.item())

        model.eval()  # режим инференса
        with torch.no_grad():  # отключаем вычисления градиентов
            for batch in tqdm(val_loader, desc=f'Validation. Epoch {epoch}'):
                generated = generate(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    max_sequence_len=max_length
                )

                val_rouge = evaluate_func(
                    generated,
                    batch['labels'].tolist(),
                    tokenizer
                )
                rouge_metrics.append(val_rouge['rouge1'])

        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | ROUGE: {np.mean(rouge_metrics):.4f}")