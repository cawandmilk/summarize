import torch
import transformers

import argparse
import logging
import os
import pprint
import tqdm

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List

from src.dataset import GPTAbstractiveSummarizationDataset
from src.metrics import get_rouge_fn
from train import get_tokenizer_and_model


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--test",
        type=str,
        nargs="+",
        # default=[
        #     "data/book/test",
        #     "data/paper/test",
        # ],
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--model_fpath",
        type=str,
        default=None,
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="gogamza/kobart-base-v1",
        help=" ".join(
            [
                "The pretrained model to use.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--save_to",
        type=str,
        default="submission",
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--inp_max_len",
        type=int,
        default=512,
        help=" ".join(
            [
                "A value for slicing the input data.",
                "It is important to note that the upper limit is determined",
                "by the embedding value of the model you want to use.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--tar_max_len",
        type=int,
        default=160,
        help=" ".join(
            [
                "A value for slicing the output data. It is used for model inference.",
                "if the value is too small, the summary may be truncated before completion.",
                "Default=%(default)s",
            ]
        ),
    )

    p.add_argument(
        "-d",
        "--debug",
        action="store_true",  ## default: False
        help=" ".join(
            [
                "Specifies the debugging mode.",
                "Default=%(default)s",
            ]
        ),
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format)


def get_datasets_for_gpt(config, tokenizer, file_dirs: List[str], max_len: int, mode: str = "train"):
    return GPTAbstractiveSummarizationDataset(
        tokenizer=tokenizer,
        file_dirs=file_dirs,
        max_len=max_len,
        mode=mode,
    )


def save_predictions(
    config: argparse.Namespace, outputs: List[dict], file_path: str
) -> Path:
    ## Save it.
    Path(config.save_to).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(file_path, encoding="utf-8", index=False)


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))

    print_config(config)

    ## Define logger.
    define_logger(config)

    if config.model_fpath != None and config.pretrained_model_name == None:
        ## Load the latest model and configuration.
        saved_data = torch.load(config.model_fpath, map_location="cpu")

        latest_checkpoint = saved_data["model"]
        train_config = saved_data["config"]

        # tokenizer = saved_data["tokenizer"]
        tokenizer, model = get_tokenizer_and_model(train_config)
        LOGGER.info(f"Tokenizer and model loaded: {train_config.pretrained_model_name}")

        model.load_state_dict(latest_checkpoint)
        LOGGER.info(f"Latest model checkpoint loaded: {config.model_fpath}")

    elif config.model_fpath == None and config.pretrained_model_name != None:
        # tokenizer = saved_data["tokenizer"]
        tokenizer, model = get_tokenizer_and_model(config.pretrained_model_name)
        LOGGER.info(f"Tokenizer and model loaded: {config.pretrained_model_name}")

    else:
        raise AssertionError()

    ## Get test dataset and generate a dataloader.
    ts_ds = get_datasets_for_gpt(
        config,
        tokenizer,
        file_dirs=config.test,
        max_len=config.tar_max_len,
        mode="test",
    )
    ts_loader = torch.utils.data.DataLoader(
        ts_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    ## Get a rouge scoreing function.
    rouge_fn = get_rouge_fn(tokenizer, is_gpt=True)

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
                labels = mini_batch["labels"]

                ## Upload to cuda.
                input_ids = input_ids.to(device)

                ## Generate.
                output = model.generate(
                    input_ids,
                    max_length=input_ids.size(0) + config.tar_max_len,  ## maximum summarization size
                    min_length=input_ids.size(0),  ## minimum summarization size
                    early_stopping=True,  ## stop the beam search when at least 'num_beams' sentences are finished per batch
                    # num_beams=config.beam_size,                 ## beam search size
                    bos_token_id=tokenizer.bos_token_id,  ## <s> = 0
                    eos_token_id=tokenizer.eos_token_id,  ## <\s> = 1
                    pad_token_id=tokenizer.pad_token_id,  ## 3
                    # length_penalty=config.length_penalty,       ## value > 1.0 in order to encourage the model to produce longer sequences
                    no_repeat_ngram_size=config.no_repeat_ngram_size,  ## same as 'trigram blocking'
                    top_k=config.top_k,
                    top_p=config.top_p,
                )


                ## Before decoding, get rouge score first.
                rouges.append(
                    rouge_fn(
                        predictions=output.cpu().detach().numpy(),
                        labels=labels,
                    )
                )

                ## Decode texts to save.
                input_ids = tokenizer.batch_decode(
                    input_ids.cpu().detach().numpy(), skip_special_tokens=True
                )
                output = tokenizer.batch_decode(
                    output.tolist(), skip_special_tokens=True
                )

                labels = labels.detach().numpy()
                labels = np.where(
                    labels == -100, tokenizer.pad_token_id, labels
                )  ## replace ignore_index (=-100) to pad token to ignore it.
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                ## Get all.
                outputs.extend(
                    [
                        {
                            "passage": input_id,
                            "answer": label,
                            "prediction": output_,
                        }
                        for input_id, label, output_ in zip(input_ids, labels, output)
                    ]
                )

                ## Update progressbar.
                pbar.update(len(output))

        ## Save answers to csv.
        if config.model_fpath != None:
            save_to = Path(
                config.save_to, config.model_fpath.split(os.path.sep)[-2] + ".csv"
            )
        else:
            save_to = Path(
                config.save_to,
                config.pretrained_model_name.replace("/", "_") + ".csv",
            )
        save_predictions(
            config,
            outputs=outputs,
            file_path=save_to,
        )
        LOGGER.info(f"Predictions saved to {save_to}")

        ## Get average of rouge scores.
        rouge = {
            key: np.mean([batch_rouge[key] for batch_rouge in rouges])
            for key in rouges[0].keys()
        }
        LOGGER.info(
            "[ROUGES] "
            + ", ".join([f"{key}: {value:.3f}" for key, value in rouge.items()])
        )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
