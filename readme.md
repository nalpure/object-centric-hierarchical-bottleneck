# Frame-Stacked Structured World Models (FS-SWM)

## Project Overview

This repository extends the work of Collu et al.'s Slot-Structured World Models (SSWM) by incorporating implicit object features using frame stacking. The primary goal is to use multiple consecutive frames to (1) train a slot attention model and (2) train a Graph Neural Network (GNN). The project is a work in progress, and the current focus is on completing step 1 (training the slot attention model).

## Training Data

The training data is generated using the 3-body physics simulation environment from Kipf et al. This environment simulates a system of three colored balls evolving according to classical gravitational dynamics. The simulation steps are computed without any external actions.

For more details on the 3-body environment, see the original repository: https://github.com/tkipf/c-swm.

## Dataset Generation

To generate datasets for training and evaluation, run the following commands:

```bash
python -m data_gen.physics --num-episodes 1000 --fname data/balls_2frame_train.h5 --seed 1
python -m data_gen.physics --num-episodes 200 --fname data/balls_2frame_eval.h5 --eval --seed 2
```

## Training Slot Attention

To train the slot attention model on single-frame data, use the following command:

```bash
python -m slot_attention.train --config single_frame
```

## Evaluating Slot Attention Model

To evaluate the performance of the trained slot attention model, run:

```bash
python -m slot_attention.evaluate --config single_frame
```