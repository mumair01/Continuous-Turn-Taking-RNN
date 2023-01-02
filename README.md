# Continuous Model of Turn-Taking using LSTM RNNs

## About

This project is my implementation of a continuous predictive model for turn-taking based on Skantze's 2017 [paper](https://www.diva-portal.org/smash/get/diva2:1141130/FULLTEXT01.pdf). The goal is to use this implementation to reproduce the results achieved in the original model and use it as a baseline for future improvements.

## Overview

Smooth turn-taking is an important aspect of natural conversation. It allows interlocutors to maintain adequate mutual comprehensibility and ensures the sequential dependency of successive turns. Humans in natural conversation are highly adept at coordinating turns by minimize gaps and overlaps while following the “one-speaker-at-a-time" rule. A consequence is that the timing between utterances is normatively constrained, and deviations convey socially relevant paralinguistic information. However, the smooth exchange of messages without communication of unintended paralinguistic information continues to be a challenge for spoken dialogue systems (SDS).

 Most dialogue systems operate on and process IPUs sequentially, one module at a time. This causes a number of issues for incorporating naturalistic patterns of turn-taking. First, modules must be applied one at a time, each with its own processing time, which adds to the response delay, and with no opportunities for self-repair i.e, the ability to correct itself. Second, it is impossible for the system to project turn completions since decision making is reactive i.e, occurs after an IPU has been detected. Third, it is impossible for the system to determine opportunities for backchannels, overlapping talk, and interruptions. Incremental systems, on the other hand, operate on small units of conversation (e.g, fixed-sized time intervals, word-level etc.) that are passed between modules.

Incremental and continuous models of end of turn and TRP detection have traditionally been challenging to implement. One key obstacles is the lack of consensus on the timescale at which these models should operate, whether there should be multiple timescales, or what the ideal incremental unit might be. Another hurdle is the lack of open-source incremental SDS frameworks, which has pushed researchers to simulate incremental input. Additionally, while individuals models of endpointing have achieved high accuracy, these may not always operate within the constraints of larger robotics systems. Despite these challenges, there has been significant progress in continuous modeling. Skantze et al. (2017) developed a generalized LSTM based model of turn-taking trained on dialogue from the MapTask corpus. In this project, we reimplement this model to use as a baseline for future models.

## Built with

Here are some of the popular frameworks this project uses:

- [Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [ML Flow](https://mlflow.org/)
- [Hydra zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)

## Getting Started

### Structure

The following is the structure for this repository.

```txt
|-- bin/
|-- data/
    |-- raw/
    |-- processed/
    |-- cached/
|-- notebooks/
|-- src/
    |-- configs/
    |-- data_lib/
    |-- experiments/
    |-- models/
    |-- *.py
|-- tests/
|-- LICENSE
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- setup.py
```
Here is a description of what each directory contains:
| Directory      | Description |
| ----------- | ----------- |
| bin      | Contains various shell scripts   |
| data   | Contains raw, processed, and cached forms of the underlying dataset |
| notebooks   | Proof of concept notebooks for the project |
| src   | Contains the data library, model, and experiment implementations |
| tests   | Pytest testing folder        |

### Usage

The first step is to clone this repository using:

```bash
git clone https://github.com/mumair01/GPT-Monologue-to-Dialogue.git
```

Next, use [conda](https://docs.conda.io/en/latest/) to install all required libraries using:

```bash
conda env create -f environment.yml
```

## Datasets

The original paper used the [HCRC MapTask](https://groups.inf.ed.ac.uk/maptask/) corpus for all experiments. The HCRC Map Task Corpus is a set of 128 dialogues that has been recorded, transcribed, and annotated for a wide range of behaviours, and has been released for research purposes. It was originally designed to elicit behaviours that answer specific research questions in linguistics.

Using the MapTask corpus directly requires an extensive knowledge of the underlying dataset structure and conventions. To abstract the dataset, we use the [data pipelines project](https://github.com/mumair01/Data-Pipelines), which allows users to load multiple datasets useful for research in conversational AI, spoken dialogue systems, and linguistics.

Since Experiment 4.1 and 4.2 require different types of data, we create two Datasets respectively: MapTaskVADDataset and MapTaskPauseDataset.

Additionally, both datasets require processing and reading large sequences per index to feed in RNNs. Therefore, we use [HDF5](https://www.google.com/search?q=h5py&sxsrf=ALiCzsZtFNnl14BXbvvlKpDLY0dytgkbwQ%3A1672664067101&ei=A9SyY-TmBcWr5NoP2uyq6AM&ved=0ahUKEwik9LCc96j8AhXFFVkFHVq2Cj0Q4dUDCBA&uact=5&oq=h5py&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIICAAQgAQQsAMyCggAEIAEELADEAoyCAgAEIAEELADMggIABCABBCwAzIHCAAQHhCwAzIJCAAQBRAeELADSgQIQRgBSgQIRhgAUM4BWOkLYPUMaAFwAHgAgAEAiAEAkgEAmAEAoAEByAEGwAEB&sclient=gws-wiz-serp) files to prepare and cache the underlying data once. This allows subsequent runs of an experiment with the same configuration to avoid the data generation process.

### MaptaskVADataset

This dataset is used to train the models in Experiment 4.1. Each index returns an x and y, where x is an array of shape (Num sequences, Num features) and y is an array of shape (Num Target Frames).

Each row in x represents a set of features at a particular timestep, both for speaker 1 and speaker 2 (in that order). There are two feature sets used: the Full set, which contains 130 features and the Prosody set, which contains 12 features. For each x, the y represents the target output voice activity from time step t to t+N. This allows the model to learn to use sequential features to predict voice activity across a future horizon.

### MapTaskPauseDataset

This dataset is used in Experiment 4.2 to evaluate the models trained using the MapTaskVADataset. It is constructed by determining the silences in each conversation. These silences (or pauses) have to have a minimum duration and exclude situations in which there is overlap before or after the silence. Therefore, each pause can be assigned a turn hold or turn shift label.

Each x in the dataset consists of a sequence of context frames before the silence from the perspective of both speaker 1 and speaker 2. Models trained for each speaker as the target speaker are then used to predict future voice activity, and the larger average prediction is selected as the next speaker. Each y contains the previous speaker, the hold or shift label, and the true next speaker.

## Experiments

In the original paper, there were three experiments in sections 4.1, 4.2, and 4.3. In this project, we implement experiments 4.1 and 4.2. We do not implement experiment due to unavailability of data.

In Experiment 4.1, we use the MaptaskVADataset to train the models to predict voice activity for a fixed future horizon from the perspective of one speaker in a dialogue.

In Experiment 4.2, we use the the trained models from the perspective of both speakers to predict the next speaker once a pause has been identified in a conversation.

## Acknowledgements

This project is a reimplementation of the paper [Towards a General, Continuous Model of Turn-taking in Spoken DialogueusingLSTM Recurrent Neural Networks](https://aclanthology.org/W17-5527.pdf).
