![Synflownet-Boltz](docs/sfn_boltz2_title.png)

[![Python versions](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![SynFlowNet paper](https://img.shields.io/badge/SynFlowNet_paper-arXiv-C70039)](https://arxiv.org/abs/2405.01155)
[![Boltz2 paper](https://img.shields.io/badge/Boltz2_paper-bioRxiv-581845)](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)

This repository contains the code for running generative screens with SynFlowNet and Boltz-2. Combining these two models allows to search the chemical space for diverse and synthesizable compounds that yield high binding affinity scores according to Boltz-2 predictions.

# Purpose

> [!NOTE] 
> **The main purpose of this repository is to offer a simple interface between SynFlowNet and Boltz-2 for running generative screens.**

SynFlowNet is a GFlowNet model that generates molecules from chemical reactions and available building blocks. A SynFlowNet model is trained on a reward function to learn to sample synthesisable molecules with a probability proportional to their reward. Here we focus on **training SynFlowNet models using Boltz-2 as a reward function**. The current repository provides features allowing to leverage such computationally expensive reward functions. It extends the original [synflownet-repo](https://github.com/mirunacrt/synflownet) codebase, itself built upon the [recursionpharma-gflownet-repo](https://github.com/recursionpharma/gflownet).

![Synflownet-Boltz](docs/sfn_boltz2_diagram.png)

To use SynFlowNet with computationally *less* expensive reward functions, it might be more advisable to simply start from the original [synflownet-repo](https://github.com/mirunacrt/synflownet). To use Boltz-2 for a different purpose such as screening a fixed molecular library, please refer to the [boltz-repo](https://github.com/jwohlwend/boltz).

# Installation

Running SynFlowNet-Boltz screens requires installing two separate environments:

1. A `boltz-env` for running the Boltz-2 workers. Start by installing boltz from the [Boltz-2 repository](https://github.com/jwohlwend/boltz) at commit `8b1627c` and add the `medchem` and `lilly-medchem-rules` packages:

```
git clone https://github.com/jwohlwend/boltz.git
cd boltz && git checkout 8b1627c
pip install -e .

pip install medchem
conda install lilly-medchem-rules
```

2. A `synflownet-env` for running the SynFlowNet trainer. This package must be installed from the current repository:
```
conda create -n synflownet-env python=3.10
pip install -e . --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

Optionally, you can install the development environment instead:
```
pip install -e '.[dev]' --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

For formatting and linting, run the following command:
```
pre-commit run --all-files
```

# Architecture Overview

## Project Structure

```
synflownet-boltz/
├── synflownet/                        # Main package
│   ├── trainer.py                     # Training orchestration
│   ├── online_trainer.py              # Standard online trainer
│   ├── algo/                          # GFlowNet algorithms
│   │   ├── trajectory_balance.py      # TB/SubTB/DB implementations
│   │   ├── soft_q_learning.py         # SQL algorithm
│   │   └── reaction_sampling.py       # Trajectory sampling
│   ├── models/                        # Neural architectures
│   │   └── graph_transformer.py       # Graph Transformer model
│   ├── envs/                          # Environments
│   │   └── synthesis_building_env.py  # Reaction template environment
│   ├── data/                          # Data management
│   │   ├── replay_buffer.py           # In-memory replay buffer
│   │   ├── async_sql_databases.py     # SQLite async storage
│   │   └── data_source.py             # Multi-source data iterator
│   ├── tasks/                         # Task definitions
│   │   └── async_reward_task.py       # Async reward task
│   └── utils/                         # Utilities
│       ├── conditioning.py            # Temperature conditioning
│       ├── synthesis_utils.py         # Reaction utilities
│       └── synthesis_evals.py         # Evaluation metrics
│
└── synflownet-boltz-launcher/         # Distributed training
    ├── configs/                       # Training configurations
    │   └── TYK2_config.yaml          # Example config
    ├── scripts/
    │   ├── train_synflownet.py       # Main training script
    │   └── boltz_reward_worker.py    # Boltz-2 reward worker
    └── data/
        └── msa_files/                # MSA files for proteins
```

## Core Components

### 1. Training System
- **GFNTrainer** (`trainer.py`): Orchestrates the training loop, manages data loaders, handles checkpointing and logging
- **AsyncRewardTrainer** (`tasks/async_reward_task.py`): Extends training for asynchronous reward computation
- **Data Flow**:
  ```
  SynFlowNet Model → Reward Queue (SQL)
         ↓
  Boltz-2 Workers (async)
         ↓
  Persistent Replay Buffer (SQL)
         ↓
  Training Batch → Parameter Update
  ```

### 2. GFlowNet Algorithms
- **Trajectory Balance (TB)** (`algo/trajectory_balance.py`): Flow-matching loss for training generative models
  - Variants: TB, SubTB(1), Detailed Balance (DB)
  - Loss functions: MSE, MAE, Huber
  - Backward policy options: Uniform, Free, MaxEnt
- **Soft Q-Learning (SQL)** (`algo/soft_q_learning.py`): Energy-based policy learning
- **Sampling** (`algo/reaction_sampling.py`): Generates molecular trajectories using reaction templates

### 3. Neural Architecture
- **GraphTransformer** (`models/graph_transformer.py`): Graph neural network using:
  - GENConv layers for node embeddings
  - TransformerConv for multi-head attention
  - Action classification heads (Stop, React, Add Building Block)
  - Backward policy parameterization

### 4. Environment
- **ReactionTemplateEnvContext** (`envs/synthesis_building_env.py`): Manages chemical reaction environment
  - Reaction template loading and application
  - Building block management
  - Action masking (legal reactions/building blocks)
  - SMILES to graph conversions

### 5. Async Reward System
- **RewardQueue** (`data/async_sql_databases.py`): SQLite database for pending reward computations
- **PersistentReplayBuffer**: Stores computed trajectories with rewards
- **Boltz-2 Workers** (`scripts/boltz_reward_worker.py`): Distributed workers computing binding affinity

### 6. Configuration
All components are configured through nested dataclasses:
- **Config** (`config.py`): Root configuration
- **AlgoConfig**: Algorithm hyperparameters (TB/SQL settings)
- **ModelConfig**: Neural network architecture
- **ReplayConfig**: Replay buffer and async databases
- **TasksConfig**: Task-specific settings (building blocks, templates)

## Key Features

1. **Asynchronous Training**: SynFlowNet trains with computationally expensive rewards (Boltz-2) without prohibitive latency
2. **Off-Policy Learning**: Uses replay buffers to reuse past trajectories
3. **Distributed Architecture**: Multiple Boltz-2 workers compute rewards in parallel
4. **SQL-Based Buffers**: Thread-safe concurrent access to shared databases
5. **Temperature Conditioning**: Controls exploration vs exploitation
6. **Reaction Templates**: Ensures synthesizability of generated molecules

# Training Guide

## Overview

Training SynFlowNet-Boltz involves running two types of processes:
1. **SynFlowNet Trainer**: Generates molecules and updates the model
2. **Boltz-2 Workers**: Compute binding affinity rewards asynchronously

## Training Architecture

The training follows an asynchronous off-policy paradigm:

```
┌─────────────────┐
│ SynFlowNet      │ Generates trajectories
│ Trainer         │ ──────────────────────┐
└─────────────────┘                       │
                                          ▼
                                 ┌──────────────────┐
                                 │ Reward Queue     │
                                 │ (SQL Database)   │
                                 └──────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
            │ Boltz-2       │    │ Boltz-2       │    │ Boltz-2       │
            │ Worker 1      │    │ Worker 2      │    │ Worker N      │
            └───────────────┘    └───────────────┘    └───────────────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                                 ┌──────────────────┐
                                 │ Replay Buffer    │
                                 │ (SQL Database)   │
                                 └──────────────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │ Training Batch   │
                                 │ Sampler          │
                                 └──────────────────┘
```

## Training Process

### Step 1: Prepare Data

#### Building Blocks
Preprocess building block molecules:
```bash
cd synflownet/data/scripts/

# Filter by molecular properties
python select_short_building_blocks.py

# Random subsampling
python subsample_building_blocks.py

# Clean and sanitize SMILES
python sanitize_building_blocks.py

# Remove duplicates
python remove_duplicates.py

# Precompute reaction-building block compatibility masks
python precompute_bb_masks.py
```

#### Protein Target

**Prepare target protein data:**
- Protein sequence (FASTA format)
- Multiple Sequence Alignment (MSA) files for Boltz-2
- Place in `synflownet-boltz-launcher/data/msa_files/`

**Generating MSA Files for Boltz-2:**

Multiple Sequence Alignments (MSA) are crucial for accurate protein structure and binding affinity prediction with Boltz-2. MSA provides evolutionary information that helps the model understand protein structure and function.

**What is MSA?**
MSA aligns homologous sequences from different organisms to identify conserved regions, which are typically functionally or structurally important. Deeper MSA (more aligned sequences) generally improves prediction confidence.

**MSA File Format:**
Boltz-2 requires MSA in A3M format, a compact alignment format where:
- Sequences are separated by `>`
- Insertions relative to query are in lowercase
- Deletions are represented by `-`

**Method 1: Using ColabFold (Recommended)**

ColabFold provides the easiest way to generate MSA files:

1. **Install ColabFold:**
   ```bash
   # Install localcolabfold
   git clone https://github.com/YoshitakaMo/localcolabfold.git
   cd localcolabfold
   ./install_colabbatch_linux.sh
   ```

2. **Generate MSA from protein sequence:**
   ```bash
   # Activate colabfold environment
   source colabfold-conda/bin/activate

   # Create FASTA file with your protein sequence
   cat > my_protein.fasta <<EOF
   >MY_PROTEIN
   TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKAD...
   EOF

   # Generate MSA (creates my_protein.a3m)
   colabfold_search my_protein.fasta /path/to/databases ./msa_output
   ```

   The output will be in `./msa_output/` directory.

3. **Using the public MMseqs2 server (no local database needed):**
   ```bash
   # ColabFold can use the public server automatically
   colabfold_batch my_protein.fasta ./predictions
   ```

   This will generate MSA files automatically in the predictions folder.

**Method 2: Using MMseqs2 Directly**

For more control or offline usage:

1. **Install MMseqs2:**
   ```bash
   conda install -c bioconda mmseqs2
   ```

2. **Download sequence databases:**
   ```bash
   # Download UniRef30 database (recommended for Boltz-2)
   mkdir databases
   cd databases
   mmseqs databases UniRef30 uniref30_db tmp
   ```

3. **Run MSA search:**
   ```bash
   # Create MMseqs2 database from your sequence
   mmseqs createdb my_protein.fasta queryDB

   # Search against UniRef30
   mmseqs search queryDB uniref30_db resultDB tmp --num-iterations 3

   # Convert to A3M format
   mmseqs result2msa queryDB uniref30_db resultDB my_protein.a3m
   ```

**Method 3: Using HHblits**

HHblits is another popular tool for generating A3M files:

1. **Install HH-suite:**
   ```bash
   conda install -c conda-forge -c bioconda hhsuite
   ```

2. **Download databases:**
   ```bash
   # Download UniRef30 or BFD
   wget https://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
   tar -xzvf UniRef30_2020_06_hhsuite.tar.gz
   ```

3. **Generate MSA:**
   ```bash
   hhblits -i my_protein.fasta \
           -d /path/to/UniRef30_2020_06 \
           -oa3m my_protein.a3m \
           -n 3 \
           -cpu 8
   ```

**Method 4: Using Boltz-2's Built-in MSA Generation**

Boltz-2 can automatically generate MSA using the MMseqs2 server:

```bash
# When running boltz predict, use the --use_msa_server flag
boltz predict input_files --use_msa_server --out_dir output
```

This will automatically query the MMseqs2 server to generate MSA for protein chains.

**Best Practices:**

1. **MSA Depth**: Aim for at least 100-1000 aligned sequences for good predictions
2. **Database Selection**:
   - UniRef30: Good balance of speed and coverage
   - BFD: More comprehensive but slower
   - MGnify: For metagenomics sequences
3. **Iterations**: Use 2-3 search iterations for better sensitivity
4. **Filtering**: Remove low-quality sequences (typically done automatically)

**Configure MSA in SynFlowNet-Boltz:**

After generating the A3M file, register it in `synflownet-boltz-launcher/data/target_to_data.yaml`:

```yaml
MY_PROTEIN:
  msa_file_path: data/msa_files/my_protein.a3m
  protein_sequence: TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKAD...
```

**Verify MSA Quality:**

Check the generated A3M file:
```bash
# Count number of sequences in MSA
grep -c "^>" my_protein.a3m

# View first few sequences
head -n 20 my_protein.a3m
```

A good MSA should have:
- At least 100+ sequences (deeper is better)
- Coverage across the entire protein length
- Diverse sequences from different organisms

**Troubleshooting:**

- **No sequences found**: Try different databases or increase search iterations
- **Too few sequences**: Use more comprehensive databases like BFD
- **File format errors**: Ensure proper A3M format with correct headers
- **Long runtime**: Use local databases instead of public servers for batch processing

### Step 2: Configure Training

Create a configuration file (e.g., `configs/my_target_config.yaml`):

```yaml
target: MY_PROTEIN_NAME
reward_queue_path: "./dbs/reward_queue.db"
persistent_replay_path: "./dbs/replay_buffer.db"
reward_cache_path: "./dbs/reward_cache.db"
reward_queue_max_size: 5000
persistent_replay_max_size: 10000000
worker_batch_size: 20
```

### Step 3: Launch Training

The training system requires running multiple processes:

#### A. Start SynFlowNet Trainer
```bash
conda activate synflownet-env
cd synflownet-boltz-launcher/scripts/

python train_synflownet.py \
  --config ../configs/my_target_config.yaml \
  --log-dir ./logs/my_run \
  --wandb-project my_project \
  --num-training-steps 10000
```

#### B. Launch Boltz-2 Workers
Start multiple workers (in separate terminals or SLURM jobs):

```bash
conda activate boltz-env
cd synflownet-boltz-launcher/scripts/

# Worker 1
python boltz_reward_worker.py \
  --config ../configs/my_target_config.yaml \
  --protein-sequence ../data/sequences/my_protein.fasta \
  --msa-dir ../data/msa_files/

# Worker 2, 3, ... (in separate processes)
python boltz_reward_worker.py ...
```

### Step 4: Monitor Training

Training progress is logged to:
- **TensorBoard**: `tensorboard --logdir ./logs/my_run`
- **Weights & Biases**: Check your wandb project dashboard
- **SQLite Logs**: Trajectory logs in `{log_dir}/train_trajectories.db`

Key metrics to monitor:
- `loss`: Trajectory Balance loss
- `logZ`: Learned partition function
- `mean_reward`: Average binding affinity
- `num_molecules_generated`: Diversity of outputs

### Step 5: Distributed Training (SLURM)

For large-scale training on HPC clusters:

```bash
# Submit trainer job
sbatch launch_trainer.sh

# Submit multiple worker jobs
sbatch --array=1-10 launch_workers.sh
```

See `synflownet-boltz-launcher/README.md` for SLURM scripts.

## Training Configuration

### Key Hyperparameters

**Algorithm Settings** (`cfg.algo`):
- `method`: "TB" or "SQL"
- `num_from_policy`: Number of on-policy samples per batch
- `num_from_dataset`: Number of dataset samples
- `max_len`: Maximum trajectory length
- `tb.variant`: "TB", "SubTB1", or "DB"
- `tb.loss_fn`: "MSE", "MAE", or "Huber"

**Model Settings** (`cfg.model`):
- `num_layers`: Graph Transformer layers (default: 3)
- `num_emb`: Embedding dimension (default: 128)
- `dropout`: Dropout rate

**Replay Buffer** (`cfg.replay`):
- `use`: Enable replay buffer
- `capacity`: Buffer size
- `warmup`: Steps before training starts
- `buffer_is_async`: Use SQL databases

**Optimization** (`cfg.opt`):
- `learning_rate`: Learning rate (default: 1e-4)
- `lr_decay`: Learning rate decay steps
- `clip_grad_param`: Gradient clipping threshold

**Temperature Conditioning** (`cfg.cond.temperature`):
- `sample_dist`: "uniform", "beta", or "gamma"
- `min`, `max`: Temperature range

### Example Configuration

```python
from synflownet.config import Config

cfg = Config()
cfg.log_dir = "./logs/my_run"
cfg.device = "cuda"
cfg.num_training_steps = 10000
cfg.validate_every = 500

# Algorithm
cfg.algo.method = "TB"
cfg.algo.num_from_policy = 16
cfg.algo.tb.variant = "SubTB1"
cfg.algo.tb.loss_fn = "Huber"

# Replay buffer
cfg.replay.use = True
cfg.replay.capacity = 100000
cfg.replay.warmup = 1000
cfg.replay.buffer_is_async = True

# Optimization
cfg.opt.learning_rate = 1e-4
cfg.opt.clip_grad_param = 10.0

# Train
from synflownet.tasks.async_reward_task import AsyncRewardTrainer
trainer = AsyncRewardTrainer(cfg)
trainer.run()
```

## Data Flow During Training

1. **Trajectory Generation**: SynFlowNet samples molecular synthesis trajectories
2. **Queue Insertion**: Molecules pushed to `RewardQueue` database
3. **Worker Processing**: Boltz-2 workers:
   - Pop batches from queue
   - Check reward cache
   - Compute binding affinity for new molecules
   - Push to `PersistentReplayBuffer`
4. **Batch Sampling**: Trainer samples from:
   - On-policy: Fresh trajectories from current model
   - Off-policy: Past trajectories from replay buffer
5. **Loss Computation**: Trajectory Balance loss computed
6. **Parameter Update**: Gradient descent step

## Checkpointing and Resuming

Checkpoints are saved every `checkpoint_every` steps to `{log_dir}/model_state.pt`:

```python
# Resume from checkpoint
cfg.start_at_step = 5000  # Resume from step 5000
trainer = AsyncRewardTrainer(cfg)
trainer.run()  # Automatically loads checkpoint
```

# Launching a screen

Please refer to [synflownet-boltz-launcher/README.md](synflownet-boltz-launcher/README.md) for instructions.

# Bibtex

If this repository is useful to your research, please consider citing the following works:

[![Boltz2 paper](https://img.shields.io/badge/Boltz2_paper-bioRxiv-581845)](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
```
@article{
passaro2025boltz, title={Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction}, author={Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Ram Somnath, Vignesh and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and others}, journal={bioRxiv}, pages={2025--06}, year={2025}, publisher={Cold Spring Harbor Laboratory}
}
```

[![SynFlowNet paper](https://img.shields.io/badge/SynFlowNet_paper-arXiv-C70039)](https://arxiv.org/abs/2405.01155)
```
@article{
cretu2025synflownetdesigndiversenovel, title={SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints}, author={Miruna Cretu and Charles Harris and Ilia Igashov and Arne Schneuing and Marwin Segler and Bruno Correia and Julien Roy and Emmanuel Bengio and Pietro Liò}, year={2025}, eprint={2405.01155}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
```
