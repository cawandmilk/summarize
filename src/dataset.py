import torch

import itertools
import json
import logging
import re
import tqdm

import numpy as np

from pathlib import Path
from typing import List


LOGGER = logging.getLogger(__name__)


def get_data(file_dir: List[str], data_type: str):
    assert data_type in ["book", "paper"]
    keys = {"input": "input", "label": "label"}

    def _get_book_data(_file_dir: List[str]) -> List[dict]:
        instances = []

        ## For every folders... (e.g., "기술과학", "기타", "사회과학", and "예술")
        for file_path in Path(_file_dir).glob("*/*.json"):
            ## Read json files.
            with open(file_path, "r", encoding="utf-8") as f:
                instance = json.load(f)

            ## Renaming and concatenate.
            instances.append(
                {
                    keys["input"]: instance["passage"],
                    keys["label"]: instance["summary"],
                }
            )

        return instances

    def _get_paper_data(_file_dir: List[str]) -> List[dict]:
        instances = []

        ## For every jsons...
        for file_path in Path(_file_dir).glob("*.json"):
            ## Read list of json files.
            with open(file_path, "r", encoding="utf-8") as f:
                instance = json.load(f)["data"]

            ## Renaming and concatenate.
            instances.extend(
                [
                    {
                        keys["input"]: instance_["summary_entire"][0]["orginal_text"],
                        keys["label"]: instance_["summary_entire"][0]["summary_text"],
                    }
                    for instance_ in instance
                ]
            )
            instances.extend(
                [
                    {
                        keys["input"]: instance_["summary_section"][0]["orginal_text"],
                        keys["label"]: instance_["summary_section"][0]["summary_text"],
                    }
                    for instance_ in instance
                ]
            )

        return instances

    ## Return.
    return (
        _get_book_data(file_dir) if data_type == "book" else _get_paper_data(file_dir)
    )


def make_clean(text):
    ## Just remove whitespaces as a space token.
    return re.sub(r"[\s]+", " ", text)


class BARTAbstractiveSummarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        file_dirs: List[str],
        max_len: int = 512,
        ignore_index: int = -100,
        mode: str = "train",
    ):
        super(BARTAbstractiveSummarizationDataset, self).__init__()

        self.tokenizer = tokenizer
        self.file_dirs = file_dirs
        self.max_len = max_len
        self.ignore_index = ignore_index
        self.mode = mode

        ## Load raw data.
        self.raw_data = list(
            itertools.chain.from_iterable(
                [
                    get_data(file_dir, data_type=Path(file_dir).parts[-2])
                    for file_dir in file_dirs
                ]
            )
        )
        LOGGER.info(f"[{self.mode}] All raw data loaded: {', '.join(file_dirs)}")

        ## Pre-process data.
        self.data = [
            {key: make_clean(value) for key, value in instance.items()}
            for instance in tqdm.tqdm(
                self.raw_data, desc="Cleaning", total=len(self.raw_data)
            )
        ]
        LOGGER.info(f"[{self.mode}] All data cleaned: {', '.join(file_dirs)}")

        ## todo: tokenize using multi-processing.
        self.data = [
            {
                key: np.array(self.tokenizer.encode(value))
                for key, value in instance.items()
            }
            for instance in tqdm.tqdm(
                self.data, desc="Tokenizeing", total=len(self.data)
            )
        ]
        LOGGER.info(f"[{self.mode}] All data tokenized: {', '.join(file_dirs)}")

    def _pad(self, text: List[int], token_id: int) -> np.ndarray:
        ## Pad or truncate.
        if len(text) < self.max_len:
            pad = np.array([token_id] * (self.max_len - len(text)))
            text = np.concatenate([text, pad])  ## pad last
        else:
            text = text[: self.max_len]  ## truncate last

        return text

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        ## Fetch the instance.
        instance = self.data[idx]

        ## Inputs for encoder.
        input_ids = self._pad(
            instance["input"], token_id=self.tokenizer.pad_token_id
        )  ## w/o special tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(float)

        ## Inputs for decoder (generator).
        decoder_input_ids = np.concatenate(
            [[self.tokenizer.eos_token_id], instance["label"]]
        )
        decoder_input_ids = self._pad(
            decoder_input_ids, token_id=self.tokenizer.pad_token_id
        )
        decoder_attention_mask = (
            decoder_input_ids != self.tokenizer.pad_token_id
        ).astype(float)

        ## Target.
        labels = np.concatenate([instance["label"], [self.tokenizer.eos_token_id]])
        labels = self._pad(labels, token_id=self.ignore_index)

        return {
            "input_ids": torch.from_numpy(input_ids).to(torch.long),
            "attention_mask": torch.from_numpy(attention_mask).to(torch.long),
            "decoder_input_ids": torch.from_numpy(decoder_input_ids).to(torch.long),
            "decoder_attention_mask": torch.from_numpy(decoder_attention_mask).to(
                torch.long
            ),
            "labels": torch.from_numpy(labels).to(torch.long),
        }
