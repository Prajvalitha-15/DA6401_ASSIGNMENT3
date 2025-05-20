
````markdown
# Assignment 3 – Sequence-to-Sequence Learning for Machine Translation

This project implements a Sequence-to-Sequence (Seq2Seq) model with LSTM-based encoder-decoder architecture for neural machine translation. The model is trained and evaluated using PyTorch, and Weights & Biases (W&B) is used for experiment tracking and logging predictions.

---

##  Objective

Build a machine translation model from a source language to a target language using:
- Encoder-Decoder RNNs (LSTM)
- Beam Search decoding
- Teacher forcing during training
- Hyperparameter tuning using W&B Sweeps

---

##  Setup

1. **Install required packages:**

```bash
pip install torch torchvision torchaudio
pip install wandb
````

2. **Login to Weights & Biases:**

```bash
wandb login
```

---

## Model Details

The model uses the following architecture:

* **Embedding Layer**
* **Multi-layer Encoder**: LSTM/GRU
* **Decoder**: LSTM with optional Beam Search
* **Dropout Regularization**
* **Cross-Entropy Loss**
* **Teacher Forcing**

Configurable hyperparameters:

* Embedding dimension
* Hidden size
* Number of layers
* Dropout rate
* Cell type (LSTM/GRU)
* Beam width
* Teacher Forcing Ratio
* Learning rate

---

## Training

Training is done via the `train_model` function with the following config:

```python
default = {
    "epochs": 20,
    "embed_dim": 64,
    "hidden_dim": 256,
    "encoder_layers": 2,
    "cell_type": "LSTM",
    "dropout": 0.2,
    "beam_width": 3,
    "learning_rate": 0.0001,
    "teacher_forcing_ratio": 0.7
}
```

To train:

```python
model = Seq2SeqRNN(...)
val_accuracy, model = train_model(model, train_loader, val_loader, default)
```

---

## Hyperparameter Tuning (W\&B Sweep)

We used Bayesian optimization to tune hyperparameters. Sweep config:

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [20]},
        'embed_dim': {'values': [64]},
        'hidden_dim': {'values': [256]},
        'encoder_layers': {'values': [3]},
        'cell_type': {'values': ['LSTM']},
        'dropout': {'values': [0.2]},
        'beam_width': {'values': [3]},
        'learning_rate': {'values': [0.0001]},
        'teacher_forcing_ratio': {'values': [0.7]}
    }
}
```

To run the sweep:

```python
sweep_id = wandb.sweep(sweep_config, project='assignment3')
wandb.agent(sweep_id, sweep_train, count=1)
```

---

##  Logging Predictions to W\&B

After training, log a few test predictions:

```python
wandb.init(project="assignment3", name="log_predictions_from_best_model")
log_predictions_to_wandb(model, test_loader, train_dataset.trg_vocab, idx_to_trg, num_samples=10)
wandb.finish()
```



## Files

```
.
├── 1234.ipynb                # Jupyter notebook with full implementation
├── README.md                 # This file
```

---

Instructions to run **W&B** in **kaggle**:
- Upload your `wandb-key` to Kaggle secrets as `wandb-key`
---
```

---

Let me know if you want me to include sample `train_model`, `log_predictions_to_wandb`, or `Seq2SeqRNN` code blocks in the README as well.
```
