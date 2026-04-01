import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from lstm_model import SequencePredictionLSTM


def generate(
        model: SequencePredictionLSTM,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
        max_sequence_len: int
) -> list[list[int]]:

    # список для хранения сгенерированных токенов
    generated_tails: list[list[int]] = []


    generated = batch['input_ids'].clone()
    lengths = batch['lengths'].clone()
    batch_size = batch['input_ids'].size(0)

    # индексы строк по порядку
    row_idx = torch.arange(batch_size)

    # маска окончания последовательности
    finished = torch.zeros(batch_size, dtype=torch.bool).cpu()

    # дополняем последовательности падингами до максимальной длины
    pad_size = max_sequence_len - generated.size(1)
    generated = F.pad(generated, (0, pad_size), value=0)

    # генерация новых токенов до тех пор, пока все
    # последовательности не будут завершены
    while ~finished.all():

        # прямой проход
        logits = model(generated, lengths)  # torch.Size([256, n_tokens, 50257])

        # токены, соответствующие логитам с максимальным значением
        # для всех элементов последовательности
        all_next_tokens = torch.argmax(logits, dim=-1)

        # позиции последних реальных токенов
        last_real_tokens_idx = lengths - 1

        # новые токены для последних элементов в последовательности
        next_tokens_for_last_element = all_next_tokens[row_idx, last_real_tokens_idx]

        # записываем новые токены к результату,
        # добавляем новые токены только к незавершенным последовательностям
        generated[row_idx[~finished], lengths[~finished]] = next_tokens_for_last_element[~finished]

        # увеличиваем длины незавершенных последовательностей на 1
        lengths[~finished] += 1

        # обновляем маску по дизъюнкции двух условий:
        # 1) сгенерирован последний токен в последовательности - EOS
        is_eos_generated: torch.BoolTensor = generated[row_idx, lengths - 1] == tokenizer.eos_token_id

        # 2) достигнута максимальная длина последовательности
        is_max_length_reached: torch.BoolTensor = lengths == max_sequence_len

        # хотя бы одно условие (ИЛИ)
        finished = is_eos_generated.cpu() | is_max_length_reached.cpu()

    # добавляем сгенерированные последовательности и результат работы функции
    for full_sequence, initial_length in zip(generated, batch['lengths']):
        generated_tails.append(full_sequence.tolist()[initial_length:])

    return generated_tails
