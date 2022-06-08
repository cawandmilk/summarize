import datasets

import numpy as np


def get_rouge_fn(tokenizer):
    ## Get a global metric.
    rouge_metric = datasets.load_metric("rouge")

    # def rouge_fn(predictions, labels):
    #     decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)

    #     return {key: value.mid.fmeasure * 100 for key, value in result.items()}

    def rouge_fn(predictions, labels):
        ## See: 
        ##  - https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/keras_callbacks#transformers.KerasMetricCallback.example
        ##  - https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        # Extract a few results
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    ## Return a function.
    return rouge_fn