import torch
import datasets
import transformers

import argparse
import datetime
import logging
import os
import pprint

from pathlib import Path
from typing import List

from src.dataset import BARTAbstractiveSummarizationDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train",
        type=str,
        nargs="+",
        default=[
            "data/book/train",
            "data/paper/train",
        ],
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--logs",
        type=str,
        default="logs",
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="ckpt",
        help=" ".join(
            [
                "Default=%(default)s",
            ]
        ),
    )

    ## Hyperparameters.
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
        "--n_epochs",
        type=int,
        default=5,
        help=" ".join(
            [
                "The number of iterations of training & validation for the entire dataset.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.2,
        help=" ".join(
            [
                "The ratio of warm-up iterations that gradulally increase",
                "compared to the total number of iterations.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help=" ".join(
            [
                "The learning rate.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help=" ".join(
            [
                "Weight decay applied to the AdamW optimizer.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--per_replica_batch_size",
        type=int,
        default=16,
        help=" ".join(
            [
                "If only 1 GPU is available, it is the same value as 'global_batch_size'.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=" ".join(
            [
                "Number of updates steps to accumulate the gradients for,",
                "before performing a backward/update pass.",
                "Default=%(default)s",
            ]
        ),
    )
    # p.add_argument(
    #     "--use_radam",
    #     action="store_false",  ## default: true
    #     help=" ".join([
    #         "Default=%(default)s",
    #     ]),
    # )
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


def get_datasets(config, file_dirs: List[str], max_len: int, mode: str = "train"):
    return BARTAbstractiveSummarizationDataset(
        tokenizer=transformers.PreTrainedTokenizerFast.from_pretrained(
            config.pretrained_model_name,
        ),
        file_dirs=file_dirs,
        max_len=max_len,
        mode=mode,
    )


def get_tokenizer_and_model(config: argparse.Namespace) -> tuple:
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        config.pretrained_model_name
    )
    model = transformers.BartForConditionalGeneration.from_pretrained(
        config.pretrained_model_name
    )

    return tokenizer, model


# def get_optimizer_and_scheduler(config: argparse.Namespace, model, n_warmup_steps: int, n_total_iterations: int) -> tuple:
#     if config.use_radam:
#         optimizer = torch.optim.RAdam(model.parameters(), lr=config.lr)
#     else:
#         ## Prepare optimizer and schedule (linear warmup and decay)
#         no_decay = ["bias", "LayerNorm.weight"]
#         optimizer_grouped_parameters = [
#             {
#                 "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#                 "weight_decay": 0.01,
#             },
#             {
#                 "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#                 "weight_decay": 0.0,
#             }
#         ]

#         optimizer = torch.optim.AdamW(
#             optimizer_grouped_parameters,
#             lr=config.lr,
#             eps=config.adam_epsilon,
#         )

#     scheduler = transformers.get_linear_schedule_with_warmup(
#         optimizer,
#         n_warmup_steps,
#         n_total_iterations,
#     )

#     return optimizer, scheduler


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))

    print_config(config)

    ## Define logger.
    define_logger(config)

    ## Get dataset.
    tr_ds = get_datasets(
        config,
        file_dirs=config.train,
        max_len=config.inp_max_len,
        mode="train",
    )
    ts_ds = get_datasets(
        config,
        file_dirs=config.test,
        max_len=config.tar_max_len,
        mode="test",
    )

    ## Get tokenizer and model.
    tokenizer, model = get_tokenizer_and_model(config)

    ## Path arguments.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.ckpt, nowtime)
    logging_dir = Path(config.logs, nowtime, "run")
    LOGGER.info(f"Output dir: {output_dir}")
    LOGGER.info(f"Logging dir: {logging_dir}")

    ## See:
    ##  - https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy="epoch",
        # prediction_loss_only=True,
        per_device_train_batch_size=config.per_replica_batch_size,
        per_device_eval_batch_size=config.per_replica_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # eval_accumulation_steps=config.eval_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.n_epochs,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        # save_total_limit=3,
        # save_steps=1000,
        fp16=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        # sharded_ddp="zero_dp_2",
        load_best_model_at_end=True,
        generation_max_length=config.tar_max_len,  ## 256
    )

    ## Define trainer.
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=ts_ds,
        tokenizer=tokenizer,
        # optimizers=get_optimizer_and_scheduler(
        #     config,
        #     model,
        #     n_warmup_steps=n_warmup_steps,
        #     n_total_iterations=n_total_iterations,
        # )
    )

    ## Train.
    trainer.train()

    ## Save the best model.
    torch.save(
        {
            "config": config,
            "tokenizer": tokenizer,
            "model": trainer.model.state_dict(),
        },
        Path(output_dir, "latest_model.pth"),
    )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
