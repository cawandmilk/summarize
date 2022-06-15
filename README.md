> 📋 A template README.md for code accompanying a Machine Learning paper

# Abstractive Summarization for Book and Paper Documents

This repository is the official implementation for the `Natural Language Processing` (GSI7625.01-00) course at Yonsei University in the first semester of 2022.

## Requirements

To install requirements:

```bash
> python -m venv venv
> source ./venv/bin/activate

> pip install -r requirements.txt
```

The "도서 자료 요약" and "논문 자료 요약" datasets are available on [AI Hub](https://aihub.or.kr/), assuming it has been rearranged into the following structure:

```bash
./data
├── [8.1M]  book
│   ├── [836K]  test
│   │   ├── [ 36K]  기타
│   │   ├── [ 84K]  예술
│   │   ├── [124K]  기술과학
│   │   └── [592K]  사회과학
│   └── [7.3M]  train
│       ├── [284K]  기타
│       ├── [612K]  예술
│       ├── [1.1M]  기술과학
│       └── [5.3M]  사회과학
└── [696M]  paper
    ├── [ 82M]  test
    │   ├── [ 52M]  논문요약_0224_0.json
    │   └── [ 30M]  논문요약_0225_6_2.json
    └── [614M]  train
        ├── [134M]  논문요약_0206_0.json
        ├── [134M]  논문요약_0206_1.json
        ├── [ 30M]  논문요약_0206_2.json
        ├── [ 47M]  논문요약_0220_0.json
        ├── [134M]  논문요약_0225_5_1.json
        └── [134M]  논문요약_0225_7_0.json

 704M used in 14 directories, 8 files
```

## Training

To train the BART model, run this command:

```bash
> python train.py \
    --train data/book/train data/paper/train \
    --logs logs \
    --ckpt ckpt \
    --pretrained_model_name gogamza/kobart-base-v1 \
    --n_epochs 5 \
    --warmup_ratio .2 \
    --lr 5e-5 \
    --weight_decay 1e-2 \
    --per_replica_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --inp_max_len 512 \
    --tar_max_len 160 \
    --debug
```

A description of all arguments can be found in [assets/train-args.txt](assets/train-args.txt).

The approximate latest checkpoint save code is as below:

```python
torch.save(
    {
        "config": config,
        "tokenizer": tokenizer,
        "model": trainer.model.state_dict(),
    },
    Path(output_dir, "latest_model.pth"),
)
```

`HuggingFace`'s log visualization is not very good, so you can check the training log in real-time by running `TensorBoard`.

```bash
> tensorboard \
    --logdir logs \
    --port {AVAILABLE-PORT}
```

Now, you can check the logs in real-time on `localhost:{AVAILABLE-PORT}`.

## Evaluation (BART)

### Evaluate Your Own Trained Model

To evaluate your model on the test dataset, run:

```bash
python eval.py \
    --test data/book/test data/paper/test \
    --model_fpath ckpt/{YOUR-MODEL-DATETIME}/latest_model.pth \
    --pretrained_model_name gogamza/kobart-base-v1 \
    --gpu_id 0 \
    --batch_size 256 \
    --no_repeat_ngram_size 3 \
    --top_k 40 \
    --top_p .95 \
    --save_to submission \
    --inp_max_len 512 \
    --tar_max_len 160 \
    --debug

python eval.py \
    --test data/book/test data/paper/test \
    --model_fpath ckpt/ours/latest_model.pth \
    --pretrained_model_name gogamza/kobart-base-v1 \
    --gpu_id 0 \
    --batch_size 256 \
    --no_repeat_ngram_size 3 \
    --top_k 40 \
    --top_p .95 \
    --save_to submission \
    --inp_max_len 512 \
    --tar_max_len 160 \
    --debug
```

A description of all arguments can be found in [assets/eval-args.txt](assets/eval-args.txt).

### Evaluate the Pre-Trained Model

Alternatively, you can try summarization using the pre-trained BART model published on `HuggingFace`:

```bash
python eval.py \
    --test data/book/test data/paper/test \
    --pretrained_model_name gogamza/kobart-base-v1 \
    --gpu_id 0 \
    --batch_size 256 \
    --no_repeat_ngram_size 3 \
    --top_k 40 \
    --top_p .95 \
    --save_to submission \
    --inp_max_len 512 \
    --tar_max_len 160 \
    --debug
```

## Evaluation (GPT)

### Evaluate the Pre-Trained Model

Also, you can try summarization using the pre-trained GPT model published on `HuggingFace`:

```bash
python eval_gpt.py \
    --test data/book/test data/paper/test \
    --pretrained_model_name skt/kogpt2-base-v2 \
    --gpu_id 0 \
    --batch_size 64 \
    --no_repeat_ngram_size 3 \
    --top_k 40 \
    --top_p .95 \
    --save_to submission \
    --inp_max_len 512 \
    --tar_max_len 160 \
    --debug
```

## Pre-trained Models

We do not publish the weights file; we publish all the code needed to reproduce the experiment.

## Results

## Quantitative Evaluation

Our model achieves the following performance on :

|                      |            | Book \[8\] |      |       | Paper \[9\] |      |       | Total \[8,9\] |      |       |
| -------------------- | :--------: | ---------: | ---: | ----: | ----------: | ---: | ----: | ------------: | ---: | ----: |
| Architecture         | Fine-tuned |         R1 |   R2 |    RL |          R1 |   R2 |    RL |            R1 |   R2 |    RL |
| KoBART V1 \[1\]      |     -      |      10.82 | 3.02 | 10.57 |       11.99 | 4.07 | 11.80 |         11.60 | 3.68 | 11.38 |
| KoBART V2 \[2\]      |     -      |      10.18 | 3.14 |  9.93 |       11.95 | 4.93 | 11.69 |         11.34 | 4.28 | 11.09 |
| KoBART-HG-SM \[3\]   |     ✓      |      13.19 | 4.05 | 12.94 |       14.76 | 5.99 | 14.56 |         14.22 | 5.30 | 14.02 |
| KoBART-SM \[4\]      |     ✓      |      14.87 | 4.64 | 14.65 |       15.83 | 6.39 | 15.66 |         15.48 | 5.74 | 15.32 |
| KoGPT2 \[5\]         |     -      |       3.01 | 0.48 |  2.95 |        2.47 | 0.56 |  2.40 |          2.66 | 0.53 |  2.60 |
| Ko-GPT-Trinity \[6\] |     -      |       4.77 | 0.99 |  4.61 |        4.32 | 0.99 |  4.16 |          4.48 | 0.98 |  4.33 |
| KoGPT \[7\]          |     -      |       6.57 | 1.52 |  6.40 |        7.20 | 2.05 |  7.04 |          6.99 | 1.85 |  5.82 |
| **Ours**             |     ✓      |      15.75 | 4.82 | 15.54 |       17.20 | 7.08 | 17.07 |         16.69 | 6.27 | 16.53 |

> We calculated the rouge scores using the `datasets` library. We are analyzing why the overall score has been leveled down. If you refer to this table, rather than comparing absolute scores, you should use it to the extent of figuring out which model is **relatively better**.

## Qualitative Evaluation

(내용 추가)

## Citation

Please cite below if you make use of the code.

```Latex
@misc{cho2022abstractive,
    title={Abstractive Summarization for Book and Paper Documents},
    author={Sang Wouk Cho and Myung Gyo Oh and Sunhyung Shim},
    year={2022},
    howpublished={\url{https://github.com/cawandmilk/summarize}},
}
```

## Reference

```plain
[1] H. Jeon, "Kobart-base-v1," https://huggingface.co/gogamza/ kobart-base-v1, 2021.
[2] H. Jeon, "Kobart-base-v2," https://huggingface.co/gogamza/kobart-base-v2, 2021.
[3] H. Jeon, "Korean news summarization model," https://huggingface.co/gogamza/kobart-summarization, 2021.
[4] S. H. Jung, "Kobart-summarization," https://github.com/seujung/KoBART-summarization, 2020.
[5] S. AI, "Kogpt2 (한국어 gpt-2) ver 2.0," https://github.com/SKT-AI/ KoGPT2, 2021.
[6] E. Davis, "Ko-gpt-trinity: Transformer model designed using sk tele- com’s replication of the gpt-3 architecture," https://huggingface.co/skt/ ko-gpt-trinity-1.2B-v0.5, 2021.
[7] I. Kim, G. Han, J. Ham, and W. Baek, "Kogpt: Kakaobrain korean(hangul) generative pre-trained transformer," https://github.com/kakaobrain/kogpt, 2021.
[8] VAIVcompany, "Book summarization dataset," https://aihub.or.kr/aidata/30713, 2021.
[9] VAIVcompany, "Paper summarization dataset," https://aihub.or.kr/aidata/ 30712, 2021.
```
