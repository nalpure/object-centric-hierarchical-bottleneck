# Frame-Stacked Structured World Models (FS-SWM)


## Training

```bash
python -m src.slot_attention.train --config full2
python -m src.slot_attention.save_slots --config full2
python -m src.explicit_latents.train --config full2
python -m src.explicit_latents.save_latents --config full2
python -m src.implicit_latents.train --config full2
```

## Evaluation
```bash
python -m src.slot_attention.eval --config full2
python -m src.explicit_latents.eval --config full2
python -m src.implicit_latents.eval --config full2
```

The configurations will be loaded from the full2.json file in the configs folder.