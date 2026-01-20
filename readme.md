# Frame-Stacked Structured World Models (FS-SWM)


## Training
To train a standard SlotAttention model for reconstruction on a Slipscape dataset run

```bash
python -m train SA --name RUN_NAME --data PATH_TO_DATASET --base 
```

To load the checkpoint with lowest total loss (a different checkpoint can be specified using the flag '--base-epoch NUMBER') and continue training with an additional contrastive and background attention loss, run
```bash
python -m train SA --name RUN_NAME_DIS --data PATH_TO_DATSET --base RUN_NAME
```

To convert the original Slipscape dataset into a dataset containing per-frame slot representations using the trained SlotAttention model, run
```bash
python -m encode_data --data PATH_TO_SLIPSCAPE_DATASET --ckpt PATH_TO_CHECKPOINT
```
The file will be saved in the same directory of the checkpoint.

To train the explicit Autoencoder for reconstruction of slots from explicit latents, run
```bash
python -m train explicit_latents --name RUN_NAME_DIS --data PATH_TO_SLOT_DATSET
```
For training with disentanglement you may use the configuration file 'explicit_latents_disent' instead.

To convert the dataset containing per-frame slot representations to per-frame explicit latents, run
```bash
python -m encode_data --data PATH_TO_SLOT_DATASET --ckpt PATH_TO_CHECKPOINT
```

To train an image predictor using per-sequence implicit latent representations, run
```bash
python -m train implicit_dynamics --name RUN_NAME_DIS --data PATH_TO_EXPLICIT_DATSET
```

## Evaluation
