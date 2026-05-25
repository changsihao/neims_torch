# NEIMS Torch

PyTorch Lightning implementation for neural electron-ionization mass spectrum prediction.

## Structure

```text
.
├── data/          # data processing scripts
├── model/         # model architectures
├── logs/          # checkpoints and training logs
├── main.py        # training entry
├── predict.py     # prediction interface and examples
└── environment.yml
```

## Installation

```bash
conda env create -f environment.yml
conda activate neims
```

## Training

```bash
python main.py
```

Checkpoints and logs are saved in `logs/`.

## Prediction

```bash
python predict.py
```

## Notes

Large dataset files (`.msp`, `.npz`, `.sdf`) are excluded from the repository.