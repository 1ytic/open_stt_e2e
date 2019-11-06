# PyTorch E2E ASR for open_stt dataset

Minimal set of scripts for training language and acoustic models for the speech recognition task.

[Russian Open Speech To Text (STT/ASR) Dataset](https://github.com/snakers4/open_stt) was taken as a data source.

## Results

Models can be trained locally, but to demonstrate the performance of scripts the [kaggle](https://www.kaggle.com) platform was choosen.

| Stage | Loss  | Updates | CER  | WER  |
|:-----:|:------|--------:|:----:|:----:|
| 1     | CTC   | 140000  | 38.1 | 82.8 |
| 2     | RNN-T | 75000   | 49.8 | 73.4 |
| 3     | RL    | 5000    | 38.1 | 66.9 |


## Requirements

- PyTorch >= 1.3 (with bug fix [#27460](https://github.com/pytorch/pytorch/pull/27460))
- [warp-rnnt](https://github.com/1ytic/warp-rnnt)
- [pytorch-edit-distance](https://github.com/1ytic/pytorch-edit-distance)


## Preprocessing

Acoustic models based on the log mel filterbanks with 40 filters of size 25ms, strided by 10ms.

- [features.py](features.py) - extract features of utterances listed in manifest file

Language model is character-based and not case sensitive.

- [utterances.py](utterances.py) - extract transcriptions of precomputed utterances


## Datasets

- [open-stt-text](https://www.kaggle.com/sorokin/open-stt-text) - transcriptions for training a language model

- [open-stt-val](https://www.kaggle.com/sorokin/open-stt-val) - validation subsets with precomputed features and transcriptions

- [open-stt-public-youtube1120-hq](https://www.kaggle.com/sorokin/open-stt-public-youtube1120-hq) - training subset with precomputed features and transcriptions


## Kernels

- [open-stt-lm](https://www.kaggle.com/sorokin/open-stt-lm) - Character-based RNN language model

- [open-stt-ctc](https://www.kaggle.com/sorokin/open-stt-ctc) - CNN-RNN acoustic model with CTC loss

- [open-stt-rnnt](https://www.kaggle.com/sorokin/open-stt-rnnt) - Character-based RNN language model and CNN-RNN acoustic model with RNN-T loss

- [open-stt-rl](https://www.kaggle.com/sorokin/open-stt-rl) - Fine-tuning with Reinforcement Learning and RNN-T loss


## Utility scripts

- [open-stt-utils](https://www.kaggle.com/sorokin/open-stt-utils) - combination of useful scripts
