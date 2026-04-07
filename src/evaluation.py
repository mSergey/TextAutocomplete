from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
from numpy import floating
from tqdm import tqdm

from src.lstm_generate import generate
from src.utils import trim_padding, trim_after_eos


def evaluate_rouge(model, tokenizer, dataloader, max_length, pad_token_id) -> floating[Any]:
    rouge_metrics: list[float] = []
    rouge = evaluate.load("rouge")
    model.eval()  # режим инференса
    with torch.no_grad():  # отключаем вычисления градиентов
        for batch in tqdm(dataloader, desc=f'Rouge evaluation...'):
            generated_tails = generate(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_sequence_len=max_length,
                pad_token_id=pad_token_id
            )

            results = rouge.compute(
                predictions=tokenizer.batch_decode(generated_tails, skip_special_tokens=True),
                references=tokenizer.batch_decode(batch['labels'].tolist(), skip_special_tokens=True),
            )

            rouge_metrics.append(results['rouge1'])
    return np.mean(rouge_metrics)


def evaluate_rouge_transformer(model, tokenizer, dataloader, max_length) -> floating[Any]:
    rouge_metrics: list[float] = []
    rouge = evaluate.load("rouge")
    max_new_tokens = int(max_length / 4)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Rouge evaluation...'):
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                use_cache=False,
                do_sample=False
            )
            input_len = batch["input_ids"].shape[1]
            generated_tails = generated[:, input_len:]
            results = rouge.compute(
                predictions=tokenizer.batch_decode(generated_tails, skip_special_tokens=True),
                references=batch['references']
            )
            rouge_metrics.append(results['rouge1'])
            break
    return np.mean(rouge_metrics)


def lstm_generate_text_examples(model, tokenizer, dataloader, max_length, pad_token_id) -> pd.DataFrame:
    input_sequences: list[str] = []
    continuations: list[str] = []
    with torch.no_grad():  # отключаем вычисления градиентов
        for batch in tqdm(dataloader, desc=f'Sample text generating...'):
            generated_tails_encoded = generate(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_sequence_len=max_length,
                pad_token_id=pad_token_id
            )

            cleaned_inputs = [
                trim_padding(seq, pad_token_id)
                for seq in batch['input_ids'].tolist()
            ]

            cleaned_generated = [
                trim_after_eos(seq, tokenizer.eos_token_id)
                for seq in generated_tails_encoded
            ]

            input_sequences.extend(tokenizer.batch_decode(
                cleaned_inputs,
                skip_special_tokens=True
            ))

            continuations.extend(tokenizer.batch_decode(
                cleaned_generated,
                skip_special_tokens=True
            ))

            samples = pd.DataFrame(
                {'Input sequence': input_sequences,
                 'Continuation': continuations}
            )
            break

    return samples


def transformer_generate_text_examples(model, tokenizer, dataloader, max_length) -> pd.DataFrame:
    input_sequences: list[str] = []
    continuations: list[str] = []
    max_new_tokens = int(max_length / 4)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Sample text generating...'):
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                use_cache=False,
                do_sample=False
            )

            input_len = batch["input_ids"].shape[1]
            inputs = generated[:, :input_len]
            generated_tails = generated[:, input_len:]

            input_sequences.extend(tokenizer.batch_decode(
                inputs,
                skip_special_tokens=True
            ))

            continuations.extend(tokenizer.batch_decode(
                generated_tails,
                skip_special_tokens=True
            ))
            break

    samples = pd.DataFrame(
        {'Input sequence': input_sequences,
         'Continuation': continuations}
    )
    return samples