# PyTorch E2E ASR for open_stt dataset

Minimal set of scripts for training language and acoustic models for the speech recognition task. Training pipeline includes the following stages:

1. Character-based RNN language model

2. CNN-RNN acoustic model with CTC loss

3. Character-based RNN language model and CNN-RNN acoustic model with RNN-T loss

4. Fine-tuning with Reinforcement Learning and RNN-T loss


## Results

The following table shows the results for [Russian Open Speech To Text (STT/ASR) Dataset](https://github.com/snakers4/open_stt).

| Stage | Model | Loss  | Updates | CER  | WER  |
|:-----:|:------|:------|--------:|:----:|:----:|
| 1     | LM    | CE    | 2407000 |      |      |
| 2     | AM    | CTC   | 216850  | 19.9 | 57.0 |
| 3     | LM+AM | RNN-T | 108425  | 21.7 | 45.6 |
| 4     | LM+AM | RL    | 300     | 19.2 | 43.9 |


## Requirements

- PyTorch >= 1.3 (with bug fix [#27460](https://github.com/pytorch/pytorch/pull/27460))
- [torch-edit-distance](https://github.com/1ytic/pytorch-edit-distance)
- [warp-rnnt](https://github.com/1ytic/warp-rnnt)


## Preprocessing

Acoustic models based on the log mel filterbanks with 40 filters of size 25ms, strided by 10ms.

- [features.py](features.py) - extract features of utterances listed in manifest file

Language model is character-based and not case sensitive.

- [utterances.py](utterances.py) - extract transcriptions of precomputed utterances


## Google Cloud Storage

Pre-trained models:

- [ru_open_stt_models](https://console.cloud.google.com/storage/browser/ru_open_stt_models)


## Kaggle Kernels

There are outdated kernels with small training subsets:

- [open-stt-lm](https://www.kaggle.com/sorokin/open-stt-lm)

- [open-stt-ctc](https://www.kaggle.com/sorokin/open-stt-ctc)

- [open-stt-rnnt](https://www.kaggle.com/sorokin/open-stt-rnnt)

- [open-stt-rl](https://www.kaggle.com/sorokin/open-stt-rl)
