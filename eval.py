import torch

import argparse
import logging
import os
import pprint
import tqdm

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List

from src.metrics import get_rouge_fn
from train import get_tokenizer_and_model, get_datasets


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_fpaths",
        required=True,
        nargs="+",
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=.95,
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--save_to",
        type=str,
        default="submission",
        help=" ".join([
            "Default=%(default)s",
        ]),
    )

    p.add_argument(
        "-d",
        "--debug",
        action="store_true",  ## default: False
        help=" ".join([
            "Specifies the debugging mode.",
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format)


def save_predictions(config: argparse.Namespace, outputs: List[dict], file_path: str) -> Path:
    ## Save it.
    Path(config.save_to).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(file_path, encoding="utf-8", index=False)


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Define logger.
    define_logger(config)

    ## For every model fpath...
    for model_fpath in config.model_fpaths:
        ## Load the latest model and configuration.
        saved_data = torch.load(model_fpath, map_location="cpu")
        
        latest_checkpoint = saved_data["model"]
        train_config = saved_data["config"]

        # tokenizer = saved_data["tokenizer"]
        tokenizer, model = get_tokenizer_and_model(train_config)
        LOGGER.info(f"Tokenizer and model loaded: {train_config.pretrained_model_name}")

        model.load_state_dict(latest_checkpoint)
        LOGGER.info(f"Latest model checkpoint loaded: {model_fpath}")

        ## Get test dataset and generate a dataloader.
        ts_ds = get_datasets(
            train_config, 
            file_dirs=train_config.test, 
            max_len=train_config.tar_max_len, 
            mode="test",
        )
        ts_loader = torch.utils.data.DataLoader(
            ts_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        ## Get a rouge scoreing function.
        rouge_fn = get_rouge_fn(tokenizer)

        ## Inference.
        with torch.no_grad():
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            device = next(model.parameters()).device

            ## Don't forget turn-on evaluation mode.
            model.eval()

            outputs = []
            rouges = []
            with tqdm.tqdm(total=len(ts_ds), desc="Inference") as pbar:

                for mini_batch in ts_loader:
                    ## Unpack.
                    input_ids = mini_batch["input_ids"]
                    attention_mask = mini_batch["attention_mask"]
                    labels = mini_batch["labels"]

                    ## Upload to cuda.
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    ## Generate.
                    output = model.generate(
                        input_ids, 
                        attention_mask=attention_mask,
                        max_length=train_config.tar_max_len,        ## maximum summarization size
                        min_length=train_config.tar_max_len // 4,   ## minimum summarization size
                        early_stopping=True,                        ## stop the beam search when at least 'num_beams' sentences are finished per batch
                        # num_beams=config.beam_size,                 ## beam search size
                        bos_token_id=tokenizer.bos_token_id,        ## <s> = 0
                        eos_token_id=tokenizer.eos_token_id,        ## <\s> = 1
                        pad_token_id=tokenizer.pad_token_id,        ## 3
                        # length_penalty=config.length_penalty,       ## value > 1.0 in order to encourage the model to produce longer sequences
                        no_repeat_ngram_size=config.no_repeat_ngram_size,   ## same as 'trigram blocking'
                        top_k=config.top_k,
                        top_p=config.top_p,
                    )

                    ## Before decoding, get rouge score first.
                    rouges.append(rouge_fn(
                        predictions=output.cpu().detach().numpy(),
                        labels=labels,
                    ))

                    ## Decode texts to save.
                    input_ids = tokenizer.batch_decode(input_ids.cpu().detach().numpy(), skip_special_tokens=True)
                    output = tokenizer.batch_decode(output.tolist(), skip_special_tokens=True)

                    labels = labels.detach().numpy()
                    labels = np.where(labels == -100, tokenizer.pad_token_id, labels) ## replace ignore_index (=-100) to pad token to ignore it.
                    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    ## Get all.
                    outputs.extend([{
                        "passage": input_id, 
                        "answer": label, 
                        "prediction": output_,
                    } for input_id, label, output_ in zip(input_ids, labels, output)])

                    ## Update progressbar.
                    pbar.update(len(output))

        ## Save answers to csv.
        save_to = Path(config.save_to, model_fpath.split(os.path.sep)[-2] + ".csv")
        save_predictions(
            config, 
            outputs=outputs, 
            file_path=save_to,
        )
        LOGGER.info(f"Predictions saved to {save_to}")

        ## Get average of rouge scores.
        rouge = {key: np.mean([batch_rouge[key] for batch_rouge in rouges]) for key in rouges[0].keys()}
        LOGGER.info("[ROUGES] " + ", ".join([f"{key}: {value:.3f}" for key, value in rouge.items()]))


if __name__ == "__main__":
    config = define_argparser()
    main(config)