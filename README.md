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

|     |               | Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :-: | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1  | Passage       | 7) 실시계약의 기간 일곱째 요소는 '특허권의 존속기간과 실시권의 기간'이다. 대개는 계약기간이 길수록 실시료가 높을 것이지만 항상 그런 것은 아니다. 어떤 기술들은 상대적으로 짧은, 특허기간보다 훨씬 짧은 존속기간을 가지기 때문이며, 따라서 생각보다 이 요소가 중요하지 않을 수 있다. 8) 특허제품의 수익성과 인기 여덟째 요소는 '특허발명의 실시로 제조된 제품의 확립된 수익성 및 그 사업상의 성공도, 현재의 인기도'이다. 미국의 판례는 상업적 성공이라는 논점을 다루면서 문제는 특허청구된 대상이라는 점을 명확하게 하여 왔다. 그러나 침해품에 따라서는 확립된 수익성이 존재하지 않는 경우도 있을 수 있고, 상품의 시장개척 초기 과정에서는 수익성을 희생하는 경우도 있을 수 있다. |
|  1  | Ground Truth  | 특허제품의 유용성과 유리함은 유사한 결과를 실현하기 위해 사용되는 구 방식이나 기기에 대한 특허제품의 실용성 및 이점으로 합리적 실시료는 특허발명이 실시될 때 얻어지는 이득에 기초하여야 한다는 관념을 반영하고 만일 협상이 깨진다면 실시료 상당액은 실시료 약정에 응할 이유를 가져야 하는 경제적 가치의 할당된 비율을 나타내야만 한다는 것이다.                                                                                                                                                                                                                                                                                                                                  |
|  1  | KoBART-SM [4] | 7) 실시계약의 기간 일곱째 요소는 '특허권의 존속기간과 실시권의 기간'으로 특허제품의 수익성과 인기 여덟째 요소로 특허발명의 실시로 제조된 제품의 확립된 수익성 및 그 사업상의 성공도, 현재의 인기도' 이다.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|  1  | KoGPT [7]     | 이 요소들은 특허권자가 특허권을 실시하는 것을 정당화하는 데 필요한 요소들이다. 이 요소들을 고려하는 경우, 특허권자는 특허권이 침해되지 않도록 하는 것이 중요하다.\n\n\[과제명\]\n특허권자의 실시권을 제한하는 요소들에 대한 분석\n\n\nI. 서론\n\n1. 문제의 제기\n\n2. 특허권자의 권리와 의무\n\n3. 특허권의 실시권 제한\n\n4. 특허발명의 보호범위\n\n5. 특허권자와 실시권자의 관계\n\n6. 특허실시허락계약\n\n7. 특허침해\n\n8. 특허제도의 목적\n\n9. 특허제도와 특허침해소송                                                                                                                                                                                                     |
|  1  | **Ours**      | 특허권의의 존속기간과 실시권의 기간은 특허제품의 수익성과 인기로, 특허청구된 대상인 특허제품은 확립된 수익성이 존재하지 않는 경우도 있고, 상품의 시장개척 초기 과정에서는 수익성을 희생하는 경우도 있다.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

|     |               | Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :-: | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  2  | Passage       | 금융위기 이후 국내의 경제 상황은 적극적 재정정책을 필요로 하는 국면이라는 점을 부인하기 어렵다. 경제성장률 저하, 내수부족, 복지수요, 고령화, 청년실업, 노동시장 상황 등은 정부의 강력하고 적극적인 재정정책을 필요로 한다고 보여진다. 재정건전성을 해치지 않는 제약조건하에서 적극적 재정정책이 요구된다면 선택할 수 있는 정책수단은 증세를 우회할 수 없다. 2016년 하반기의 국가채무의 증가 전망이 2016년 초에 예측한 것보다 낮게 나타난 것은 연초의 예측보다 2016년의 세수입이 높게 실현된 것에 기인한다. 이는 세법개정을 통한 노력에 세수증대의 노력에 의한 것이 아니라 담배소비에 대한 예측을 낮게 하여 예측대비 높게 실현된 담배소비와 이로 인한 담배소비세 수입, 부동산 경기 일시적 활성화에 |
|  2  | Ground Truth  | 금융위기 이후 강력하고 적극적인 재정정책이 필요하다. 재정건전성을 해치지 않는 재정정책은 증세가 대표적인데, 2016년 하반기의 국가채무 전망이 낮은 것도 담배소비세와 양도세 수입으로 세금이 늘어났기 때문이다.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|  2  | KoBART-SM [4] | 금융위기 이후 국내의 경제 상황은 적극적 재정정책을 필요로 하는 국면이라는 점을 부인하기 어렵지만, 경제성장률 저하, 내수부족, 복지수요, 고령화, 청년실업, 노동시장 상황 등은 정부의 강력하고 적극적인 재정정책을 요구한다고 보여진다.                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|  2  | KoGPT [7]     | 2016년 하반기 국가채무가 증가한 것은 2016년 초의 예측보다 세수입의 증가가 높게 나타난 데 기인한다\n\[사업명\]\n한국과학기술기획평가원연구운영비지원\n\n\[과제명\]: 2016년 하반기 경제전망\n\n\n사업\_부처명: 미래창조과학부\n\n분류: NaN\[요약문\_연구목표\]\n본 연구는 기존의 연구에서 다루지 않았던, 뇌의 구조적 변화와 인지기능의 관계를 규명하고자 한다. 이를 위해, 뇌 구조의 변화를 유발하는 뇌 손상, 뇌 손상 후 발생하는 인지기능 저하, 인지기능 저하의 회복 과정을 추적 관찰하고자 하며, 인지 기능의 회복 과정에서 뇌 구조적 기능의 변화가 수반                                                                                                                                            |
|  2  | **Ours**      | 금융위기 이후 국내의 경제 상황은 적극적 재정정책을 필요로 하는 국면이다. 경제성장률 저하, 내수부족, 복지수요, 고령화, 청년실업, 노동시장 상황 등은 정부의 강력하고 적극적인 재정정책을 요구한다.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

|     |               | Text                                                                                                                                                                                                                                                                                                                                                                                                        |
| :-: | :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  3  | Passage       | STEC의 배양 배지로 SMAC만은 불충분하며 여기에 cefixime, vancomycin, novobiocin 등 항균제와 potassium tellurite 및 bile salt 등을 첨가시키면 STEC를 제외한 장내세균들을 효과적으로 억제시킬 수 있다는 보고가 있으며, 본 연구에서는 Fukushima 등이 사용한 방법인 SMAC 배지에 cefixime과 potassium tellurite를 첨가한 CT-SMAC 배지에서 항균제인 cefixime의 농도를 재조정하여 배양배지로 사용하였다. 즉 본 연구 |
|  3  | Ground Truth  | 본 연구에서는 Fukushima 등이 사용한 방법인 SMAC 배지에 cefixime과 potassium tellurite를 첨가한 CT-SMAC 배지에서 항균제인 cefixime의 농도를 재조정하여 배양배지로 사용하였으며 그 결과 cefixime 농도는 Fukushima 등이 사용했던 0.05 mg/L의 2배인 0.1 mg/L가 최적 농도임을 입증하였다.                                                                                                                        |
|  3  | KoBART-SM [4] | Fukushima 등이 사용한 방법인 SMAC 배지에 cefixime과 potassium tellurite를 첨가한 CT-SMAC 배지에서 항균제인 ceFixxim의 농도를 재조정하여 배양배지로 사용하여 STEC를 제외한 장내세균들을 효과적으로 억제시킬 수 있다는 보고가 있다.                                                                                                                                                                           |
|  3  | KoGPT [7]     | 본 연구에서 사용된 CT-STEMS는 SMAD-STEM을 기본으로 하여 SMA-STEAM으로 개량된 것으로, SMA는 S-S-M의 약자로 SMA가 SMA보다 S가 더 많이 포함되어 있다. SMA의 S는 S, M은 M, S는 M, M는 S를 의미하며, S-M-S의 SMA를 SMA로 표기한다. S-ST-M은 SMA와 STEM을 조합한 것으로 STEM은 ST-S, ST-M, STEM의 약자이다. S는 STEM, M, ST는 ST, M을 의미하는데 ST는 S와 M을, STM은 ST와 M를                                     |
|  3  | **Ours**      | SMAC 배지에 cefixime과 potassium tellurite를 첨가한 CT-SMAC 배지에서 항균제인 ceFixme의 농도를 재조정하여 배양배지로 사용하였다.                                                                                                                                                                                                                                                                            |

|     |               | Text                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :-: | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  4  | Passage       | 『圓覺經』의 교설은 일체 중생 누구나 청정한 圓覺妙心(圓覺)을 갖추고 있으나 이 ‘淸淨圓覺’이 어느 시점에서 幻에 뒤덮여 오염되어졌으며, 이 오염된 원각을 원래의 청정한 원각의 상태로 되돌아가게 하기 위하여 실천수행 할 것을 권하고 있는 경전이다. 이 『원각경』의 원명은 『大方廣圓覺修多羅了義經』이며 梵語로는 Mah?vaipulya-p?r?abuddha-s?tra-prasann?rtha-s?tra로 표기하기도 한다. 부처님께서는 根機에 따른 다양한 교설          |
|  4  | Ground Truth  | 『圓覺經』의 교설은 일체 중생 누구나 청정한 圓覺妙心을 갖추고 있으나 이 ‘淸淨圓覺’이 어느 시점에서 幻에 뒤덮여 오염되어졌으며, 이 오염된 원각을 원래의 청정한 원각의 상태로 되돌아가게 하기 위하여 실천수행 할 것을 권하고 있는 경전이다.                                                                                                                                                                                         |
|  4  | KoBART-SM [4] | 『원각경'은 '淸淨圓覺'이 어느 시점에서 幻에 뒤덮여 오염되어졌으며 이 오염된 원각을 원래의 청정한 원각의 상태로 되돌아가게 하기 위하여 실천수행 할 것을 권하고 있는 경전이며 범어로는 Mah?vaipulya-p?r?abuddha-s?tra-prasann?rtha-s.tra로 표기하기도 한다.                                                                                                                                                                         |
|  4  | KoGPT [7]     | 『원각 경』은 원각(圓角)이라는 말로 표현되는 불성(佛性)을 주제로 하여, 원각과 무명(無明)의 관계, 원각의 체(體)와 상(相)에 대하여 설하고 있다. 원각은 완전하고 청정한 깨달음의 경지를 말하며, 무명(밝지 못함)은 원각의 반대로서 미망(迷妄)의 상태를 말한다.\n\[과제\]\n원각경의 내용을 요약하고, 원각이라는 개념이 어떻게 성립되었는지를 서술하시오.\n\n\[원각 경의 내용 요약\]\n\n원각의 의미\n\n\n1. 원각의 의미 원각이란,\n\n불 |
|  4  | **Ours**      | 『圓覺經』의 교설은 일체 중생 누구나 청정한 圓覺妙心을 갖추고 있으나 이 ‘淸淨圓覺’이 어느 시점에서 幻에 뒤덮여 오염되어졌으며, 이 오염된 원각을 원래의 청정한 원각의 상태로 되돌아가게 하기 위하여 실천수행 할 것을 권하고 있는 경전이다.                                                                                                                                                                                         |

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
