import evaluate


def compute_rouge_score(generated_sequences, references, tokenizer):

    rouge = evaluate.load("rouge")

    pred_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
    ref_texts = tokenizer.batch_decode(references, skip_special_tokens=True)

    results = rouge.compute(
        predictions=pred_texts,
        references=ref_texts
    )

    return results
