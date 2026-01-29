# ColModernVBERT Training and Validation Guide

This guide provides step-by-step instructions for training and validating the ColModernVBERT model on the Tatdqa dataset.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration Setup](#configuration-setup)
4. [Training Process](#training-process)
5. [Validation Process](#validation-process)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- Git (for cloning the repository)
- Sufficient disk space for datasets and model checkpoints (~10GB minimum)

---

## Environment Setup

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv colpali_env

# Activate virtual environment
# On Windows:
colpali_env\Scripts\activate

# On Linux/Mac:
source colpali_env/bin/activate
```

### Step 2: Install Local Project

Navigate to your project directory and install the project in editable mode:

```bash
# Navigate to project directory
cd D:\Github\colpali

# Install the project with training dependencies
pip install -e ".[train]"
```

This will install:
- All base dependencies (numpy, torch, transformers, etc.)
- All training dependencies (configue, datasets, typer, accelerate, etc.)
- The project in editable mode (changes take effect immediately)

### Step 3: Verify Installation

```bash
# Check if the package is installed
python -c "import colpali_engine; print('colpali_engine installed successfully')"

# Check if training dependencies are available
python -c "import configue; import datasets; import typer; print('All training deps available')"

# Verify your local code is being used
python -c "import colpali_engine; print(colpali_engine.__file__)"
# Should show: D:\Github\colpali\colpali_engine\__init__.py
```

### Step 4: Set Environment Variable

Set the environment variable to load datasets from HuggingFace:

```bash
# On Windows (PowerShell):
$env:USE_LOCAL_DATASET="0"

# On Windows (CMD):
set USE_LOCAL_DATASET=0

# On Linux/Mac:
export USE_LOCAL_DATASET=0

# Verify it's set
# Windows PowerShell:
echo $env:USE_LOCAL_DATASET
# Windows CMD / Linux/Mac:
echo $USE_LOCAL_DATASET
# Should output: 0
```

### Step 5: Verify GPU Availability (Optional)

If using GPU, verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Configuration Setup

### Understanding the Config File

The training configuration file is located at:
```
scripts/configs/train_colmodernvbert_16gb.yaml
```

### Key Configuration Settings

#### Training Dataset
- **Filter**: `filter_by_source: "tatdqa"` - Trains only on Tatdqa dataset
- **Source**: `vidore/colpali_train_set` (loaded from HuggingFace)

#### Validation Dataset
- **Current**: `syntheticDocQA_energy_test` (default)
- **To validate on Tatdqa**: Change `eval_dataset` section (see below)

#### Training Parameters
- **Epochs**: 3
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps (effective batch size = 8)
- **Learning Rate**: 5e-5
- **Mixed Precision**: bfloat16
- **Gradient Checkpointing**: Enabled (saves memory)

#### Validation Parameters
- **Strategy**: `eval_strategy: "steps"` - Validation runs at step intervals
- **Frequency**: `eval_steps: 500` - Validation every 500 training steps
- **Enabled**: `run_eval: true` - Enables validation during training

### Update Config for Tatdqa Validation

To validate on Tatdqa dataset instead of the default, update the `eval_dataset` section in `scripts/configs/train_colmodernvbert_16gb.yaml`:

```yaml
eval_dataset:
  tatdqa:
    (): colpali_engine.utils.dataset_transformation.load_eval_set
    dataset_path: "vidore/tatdqa_test"
```

**Full updated config section:**

```yaml
config:
  model:
    (): colpali_engine.models.ColModernVBert.from_pretrained
    pretrained_model_name_or_path: "ModernVBERT/colmodernvbert"
    torch_dtype: "bfloat16"
    device_map: "cuda:0"
  
  processor:
    (): colpali_engine.models.ColModernVBertProcessor.from_pretrained
    pretrained_model_name_or_path: "ModernVBERT/colmodernvbert"
  
  train_dataset:
    (): colpali_engine.utils.dataset_transformation.load_train_set
    filter_by_source: "tatdqa"  # Training on tatdqa only
  
  eval_dataset:
    tatdqa:  # Changed to validate on tatdqa
      (): colpali_engine.utils.dataset_transformation.load_eval_set
      dataset_path: "vidore/tatdqa_test"
  
  tr_args:
    (): transformers.training_args.TrainingArguments
    output_dir: "./output/colmodernvbert"
    overwrite_output_dir: true
    num_train_epochs: 3
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    per_device_eval_batch_size: 1
    eval_strategy: "steps"  # Evaluation happens at step intervals
    dataloader_num_workers: 2
    bf16: true
    gradient_checkpointing: true
    save_steps: 500
    logging_steps: 10
    eval_steps: 500  # Run validation every 500 training steps
    warmup_steps: 500
    learning_rate: 5e-5
    save_total_limit: 2
  
  max_length: 256
  run_train: true  # Enable training
  run_eval: true   # Enable validation during training
  output_dir: "./output/colmodernvbert"
```

---

## Training Process

### Step 1: Navigate to Project Directory

```bash
cd D:\Github\colpali
```

### Step 2: Run Training Script

```bash
python scripts/train/train_colbert.py scripts/configs/train_colmodernvbert_16gb.yaml
```

### Step 3: Monitor Training Progress

You will see output similar to:

```
Loading config
Creating Setup
GPU Utilization: ...
Training model
{'loss': 2.345, 'learning_rate': 0.00005, 'epoch': 0.1}
{'loss': 2.234, 'learning_rate': 0.00005, 'epoch': 0.2}
...
{'train_runtime': 123.45, 'train_samples_per_second': 0.5, 'epoch': 0.5}
Running evaluation...
{'eval_loss': 1.987, 'eval_runtime': 45.67, 'eval_samples_per_second': 1.2}
...
```

### Step 4: Check Output Directory

Training checkpoints are saved in:
```
./output/colmodernvbert/
```

Directory structure:
```
./output/colmodernvbert/
├── checkpoint-500/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── checkpoint-1000/
├── checkpoint-1500/
├── training_config.yml
└── git_hash.txt
```

---

## Validation Process

### How Validation Works

With `run_train: true` and `run_eval: true`:

1. **Validation Schedule**: 
   - Validation runs every **500 training steps** (based on `eval_steps: 500`)
   - Not after each epoch, but at step intervals

2. **Validation Dataset**: 
   - Uses the dataset specified in `eval_dataset` section
   - Currently set to `tatdqa_test` (if you updated the config)

3. **Validation Metrics**:
   - `eval_loss`: Average loss on validation set
   - `eval_runtime`: Time taken for evaluation
   - `eval_samples_per_second`: Throughput

### Validation Output Example

```
Running evaluation...
{'eval_loss': 1.987, 'eval_runtime': 45.67, 'eval_samples_per_second': 1.2, 'step': 500}
```

### View Detailed Training Logs

Training logs are saved in:
```
./output/colmodernvbert/runs/
```

View with TensorBoard (optional):
```bash
pip install tensorboard
tensorboard --logdir ./output/colmodernvbert/runs
```

Then open `http://localhost:6006` in your browser.

---

## Understanding Results

### Training Metrics

- **loss**: Training loss (decreases over time)
- **learning_rate**: Current learning rate (may change with warmup)
- **epoch**: Current epoch progress (0.0 to 3.0)
- **train_runtime**: Total training time
- **train_samples_per_second**: Training throughput

### Validation Metrics

- **eval_loss**: Validation loss (lower is better)
- **eval_runtime**: Time taken for validation
- **eval_samples_per_second**: Validation throughput

### Model Checkpoints

- **checkpoint-500**: Model saved at step 500
- **checkpoint-1000**: Model saved at step 1000
- **checkpoint-1500**: Model saved at step 1500
- Only the last 2 checkpoints are kept (`save_total_limit: 2`)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'configue'"

**Solution:**
```bash
pip install -e ".[train]"
```

### Issue: "CUDA out of memory"

**Solutions:**
- Reduce `per_device_train_batch_size` to 1 (already set)
- Reduce `gradient_accumulation_steps` if needed
- Ensure `gradient_checkpointing: true` is set (already set)
- Use CPU instead: Change `device_map: "cuda:0"` to `device_map: "cpu"` (slower)

### Issue: "Dataset is empty after filtering"

**Solutions:**
- Verify `filter_by_source: "tatdqa"` matches the dataset
- Check that `USE_LOCAL_DATASET=0` is set correctly
- Verify dataset is accessible from HuggingFace

### Issue: "Config must be of type ColModelTrainingConfig"

**Solution:**
- Ensure your config file is properly formatted
- Check YAML syntax for errors
- Verify all required fields are present

### Issue: Validation not running

**Solutions:**
- Ensure `run_eval: true` is set
- Ensure `eval_dataset` is properly configured
- Check that `eval_steps` is not too large
- Verify `eval_strategy: "steps"` is set

### Issue: "FileNotFoundError" or path errors

**Solutions:**
- Ensure you're in the correct directory
- Use absolute paths if relative paths don't work
- Check that all config files exist

---

## Next Steps

### After Training Completes

1. **Locate Final Model**:
   ```
   ./output/colmodernvbert/checkpoint-XXXX/
   ```

2. **Load Trained Model for Inference**:
   ```python
   from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
   import torch

   # Load your trained model
   model = ColModernVBert.from_pretrained(
       "./output/colmodernvbert/checkpoint-XXXX",
       torch_dtype=torch.bfloat16,
       device_map="cuda:0"
   ).eval()

   processor = ColModernVBertProcessor.from_pretrained(
       "./output/colmodernvbert/checkpoint-XXXX"
   )
   ```

3. **Continue Training** (if needed):
   - Modify config to remove `filter_by_source` for multi-dataset training
   - Update `pretrained_model_name_or_path` to point to your checkpoint
   - Run training again

4. **Evaluate on Test Split**:
   - Use `vidore-benchmark` package for comprehensive evaluation
   - See README.md for benchmarking instructions

5. **Benchmark on Benchmark Split**:
   - Submit to ViDoRe leaderboard
   - Follow instructions in README.md

---

## Quick Reference

### Validation Behavior

| Config Setting | Behavior |
|---------------|----------|
| `eval_strategy: "steps"` | Validation runs at step intervals |
| `eval_steps: 500` | Validation every 500 training steps |
| `eval_strategy: "epoch"` | Validation after each epoch |
| `run_eval: true` | Enables validation during training |
| `run_train: false` | Only evaluation (training disabled) |

### Training Workflow Summary

```
1. Create virtual environment
   → python -m venv colpali_env
   → Activate it

2. Install local project
   → cd D:\Github\colpali
   → pip install -e ".[train]"

3. Set environment variable
   → export USE_LOCAL_DATASET=0 (Linux/Mac)
   → set USE_LOCAL_DATASET=0 (Windows)

4. Update config file (optional)
   → Change eval_dataset to tatdqa_test

5. Run training
   → python scripts/train/train_colbert.py scripts/configs/train_colmodernvbert_16gb.yaml

6. Monitor training
   → Watch console for validation results every 500 steps
   → Check output directory for checkpoints

7. Training completes
   → Model saved in ./output/colmodernvbert/
   → Use checkpoint for inference or further training
```

---

## Additional Resources

- **Project README**: See `README.md` for more details
- **Config Files**: Located in `scripts/configs/`
- **Training Script**: `scripts/train/train_colbert.py`
- **Original ColPali Repository**: https://github.com/illuin-tech/colpali
- **ModernVBERT Paper**: https://arxiv.org/html/2510.01149v1
- **ViDoRe Benchmark**: https://github.com/illuin-tech/vidore-benchmark

---

## Notes

- Training on Tatdqa only: Config includes `filter_by_source: "tatdqa"`
- Validation frequency: Every 500 steps (not after each epoch)
- Checkpoint saving: Every 500 steps, keeping last 2 checkpoints
- Mixed precision: Uses bfloat16 for memory efficiency
- Gradient checkpointing: Enabled to save GPU memory

---

**Last Updated**: Based on `train_colmodernvbert_16gb.yaml` configuration
