import datasets

import numpy as np


def get_rouge_fn(tokenizer, is_gpt: bool = False):
    ## Get a global metric.
    rouge_metric = datasets.load_metric("rouge")

    def rouge_fn(predictions, labels):
        ## See:
        ##  - https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/keras_callbacks#transformers.KerasMetricCallback.example
        ##  - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb

        if is_gpt:
            decoded_preds = tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_preds = [i.split("\n[요약 결과]\n")[-1] for i in decoded_preds]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        else:
            decoded_preds = tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        # Extract a few results
        result = {
            key: np.mean([value.mid.fmeasure, value.low.fmeasure, value.high.fmeasure])
            * 100
            for key, value in result.items()
        }

        # Add mean generated length
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    ## Return a function.
    return rouge_fn
