# TT-Sparse: _Learning Sparse Rule Models with Differentiable Truth Tables_

**Paper:** [ICML 2026](https://arxiv.org/pdf/2603.07606)

## Installation

```bash
pip install git+https://github.com/hansfarrell/tt-sparse.git
```

Or install locally for development:

```bash
git clone https://github.com/hansfarrell/tt-sparse.git
cd tt-sparse
pip install -e ".[test]"
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quickstart

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tt_sparse import TabularEncoder, TTSparseModel, train, prune, explain, predict_rules

# Load Pima Indians Diabetes (OpenML ID 37)
dataset = fetch_openml(data_id=37, as_frame=True)
df = dataset.frame
df = df.rename(columns={"class": "target"})

# Train/test split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# Encode continuous features into binary via thermometer encoding
encoder = TabularEncoder(target="target", task_type="binary", num_bits=9)
train_data = encoder.fit_transform(df_train)

# Build model
model = TTSparseModel(
    input_size=encoder.n_ltt_features,
    num_nodes=30,
    num_classes=1,
    n_bits=5,
    tau=0.01,
    task_type="binary",
    skip_size=encoder.n_skip_features,
)

# Train
train(model, train_data, epochs=100, device="cpu", seed=0, verbose=True)

# Prune
prune(model, train_data, max_drop_pct=2.0, finetune_epochs=30, device="cpu", seed=0, verbose=True)

# Extract human-readable rules
rules = explain(model, encoder)
print(f"\nExtracted {len(rules.rules)} rules (complexity: {rules.complexity}):\n")
for rule in rules.rules:
    print(f"  {rule['r']}")
    print(f"    w={rule['w']}\n")

# Rule inference (no neural network needed)
rule_preds = predict_rules(rules, df_test.drop(columns=["target"]), encoder=encoder)
```

Example output (complexity = 7, AUC = 0.805):

```
True                          w={'tested_positive': -1.55}
pres                          w={'tested_positive': -0.24}
mass                          w={'tested_positive':  0.41}
pedi                          w={'tested_positive':  0.23}
age                           w={'tested_positive':  0.59}
((plas >= 167.60))            w={'tested_positive':  1.48}
((mass >= 28.17))             w={'tested_positive':  0.98}
```

## Quick Docs

### `TabularEncoder`

```python
TabularEncoder(
    target="target",       # Name of the target column.
    categorical=None,      # List of categorical column names (auto-detected if None).
    continuous=None,       # List of continuous column names (auto-detected if None).
    num_bits=9,            # Number of thermometer bits per continuous feature.
    task_type="binary",    # "binary", "multiclass", or "regression".
)
```
- `.fit(df)`: Fit encoder on a DataFrame containing the target column.
- `.transform(df)`: Returns `{"X_ltt": ..., "X_skip": ..., "y": ...}`.
- `.fit_transform(df)`: Fit and transform in one step.
- `.n_ltt_features`: Number of binary features (for `input_size`).
- `.n_skip_features`: Number of skip features (for `skip_size`).

### `TTSparseModel`

```python
TTSparseModel(
    input_size,            # Number of binary features (from encoder).
    num_nodes,             # Number of truth-table neurons.
    num_classes,           # Output classes (1 for binary/regression).
    n_bits,                # Inputs per neuron (truth table has 2^n_bits entries).
    tau=0.01,              # Temperature for soft top-k relaxation.
    task_type="binary",    # "binary", "multiclass", or "regression".
    skip_size=0,           # Must match encoder.n_skip_features (0 to disable).
    dropout=0.0,           # Dropout rate on node outputs during training.
)
```

### `train()`

```python
train(
    model, data, *,
    epochs=100,            # Maximum training epochs.
    batch_size=2048,       # Mini-batch size.
    lr=0.01,               # Learning rate.
    val_split=0.2,         # Fraction held out for early stopping.
    patience=15,           # Epochs without improvement before stopping.
    device="cpu",          # "cpu" or "cuda".
    seed=0,                # Random seed for reproducibility.
    verbose=False,         # Print progress every 10 epochs.
)
```

### `prune()`

The original paper uses L1 magnitude-based pruning. This implementation extends it with saliency scoring (`|weight × gradient| / complexity_gain`) to select which connections to prune at each iteration.

```python
prune(
    model, data, *,
    max_drop_pct=0.0,      # Max metric drop (%) for accepting a pruning step.
    collapse_drop_pct=15.0,# Metric drop (%) that triggers collapse revert.
    finetune_epochs=30,    # Finetuning epochs after each pruning step.
    finetune_batch_size=2048,
    max_iterations=80,     # Maximum pruning iterations.
    max_fanin=16,          # Hard cap on inputs per node.
    device="cpu",
    seed=0,
    verbose=False,
)
```

Returns a stats dict with sparsity metrics, complexity costs, and the Pareto frontier.

### `explain()`

```python
explain(model, encoder) -> RuleSet
```

Extracts Boolean rules from the pruned model. Each truth-table node becomes a DNF rule minimized via Quine-McCluskey with don't-care optimization.

### `predict_rules()`

```python
predict_rules(rules, df, encoder=encoder) -> np.ndarray
```

Runs inference using only the extracted rules. Predictions are numerically equivalent to the pruned model.

## How It Works

1. **Encoding**: continuous features become binary via thermometer thresholds and categorical features are one-hot encoded
2. **Training**: each neuron learns which `n_bits` inputs to attend to (via differentiable top-k) and what Boolean function to compute (via a soft truth table)
3. **Pruning**: saliency-based edge removal simplifies the network while a Pareto frontier preserves the best accuracy-complexity trade-offs
4. **Rule extraction**: each neuron's truth table is enumerated, minimized with Quine-McCluskey (exploiting don't-care terms), and expressed as a DNF rule
5. **Rule inference**: the final model is a weighted sum of Boolean rules, not the neural network.

## License

MIT
