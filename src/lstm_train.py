import evaluate
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.lstm_generate import generate
from src.lstm_model import SequencePredictionLSTM

# функция обучения нейросети LSTM
def train(
        model: SequencePredictionLSTM,
        n_epochs: int,
        tokenizer: PreTrainedTokenizerBase,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: CrossEntropyLoss,
        optimizer: Adam,
        scheduler: ReduceLROnPlateau,
        max_length: int,
        pad_token_id: int
) -> None:
    rouge = evaluate.load("rouge")
    vocab_size = model.fc.out_features
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        rouge_metrics: list[float] = []
        for batch in tqdm(train_loader, desc=f'LSTM training. Epoch {epoch}'):
            inputs = batch['input_ids']
            labels = batch['labels']
            lengths = batch['lengths']
            optimizer.zero_grad()
            logits = model.forward(inputs, lengths)
            loss = criterion(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )
            # считаем градиенты
            loss.backward()
            # обновляем веса
            optimizer.step()
            total_loss += loss.item()

        model.eval()  # режим инференса
        with torch.no_grad():  # отключаем вычисления градиентов
            for batch in tqdm(val_loader, desc=f'LSTM validation. Epoch {epoch}'):
                generated_tails = generate(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    max_sequence_len=max_length,
                    pad_token_id=pad_token_id
                )

                results = rouge.compute(
                    predictions=tokenizer.batch_decode(generated_tails, skip_special_tokens=True),
                    references=tokenizer.batch_decode(batch['labels'], skip_special_tokens=True),
                )

                rouge_metrics.append(results['rouge1'])

        scheduler.step(np.mean(rouge_metrics))

        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | ROUGE: {np.mean(rouge_metrics):.4f}")