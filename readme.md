# Frame-Stacked Structured World Models (FS-SWM)

## Project Overview

This repository extends the work of Collu et al.'s Slot-Structured World Models (SSWM) by incorporating implicit object features using frame stacking. The primary goal is to use multiple consecutive frames to (1) train a slot attention model, (2) train a Graph Neural Network to predict future frames. A disentenglement loss will be applied by perturbing a single object feature (explicit or implicit) in the first frame and rolling out the following frames accordingly. The project is a work in progress.

## Training Data

The training data is generated using the 3-body physics simulation environment from Kipf et al. This environment simulates a system of three colored balls evolving according to classical gravitational dynamics. The simulation steps are computed without any external actions.

For more details on the 3-body environment, see the original repository: https://github.com/tkipf/c-swm.

## Dataset Generation

To generate datasets for training and evaluation, run the following commands:

```bash
python -m data_gen.physics --stacked-frames 5 --num-episodes 1000 --fname data/5F_1000ep_s1 --seed 1
python -m data_gen.physics --stacked-frames 5 --num-episodes 200 --fname data/data/5F_200ep_s2 --eval --seed 2
```

## Option 1: Invididual training of SA and GNN

### Training SA model

To train the slot attention model, use the following command:

```bash
python -m slot_attention.train --config 5frame
```

The configuration will be loaded from the configs.json file.


### Optional: Evaluating SA model

To evaluate the performance of the trained slot attention model, run:

```bash
python -m slot_attention.eval --config 5frame
```

The configuration will be loaded from the configs.json file.

### Training GNN model

To load the SA model and train a GNN model on it, use the following command: 

```bash
python -m train --config 5frame
```

The configuration will be loaded from the sswm_configs.json file.

## Option 2: End-to-end training

To train the model end-to-end, use the following command:

```bash
python -m train --config 5frameEndToEnd
```

The configuration will be loaded from the sswm_configs.json file.

## Evaluation

To evaluate the final model performance, use the following command:

```bash
python -m eval --config 5frameEndToEnd
```

The configuration will be loaded from the sswm_configs.json file.