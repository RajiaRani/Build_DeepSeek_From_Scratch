# Running vLLM on your university HPC (with VS Code)

A step by step guide to run vLLM on a GPU node of your university cluster, and to drive it
from VS Code. Commands assume a **SLURM** scheduler, which almost all clusters use.

## The one thing to understand first

- The **login node** is just a doorway you SSH into. It has no GPU for you. Never run
  models there.
- You ask SLURM for a **compute node** with a GPU, and run vLLM there. Two ways to get one:
  a **batch job** (`sbatch`, fire and forget) or an **interactive session** (`srun`, a live
  shell). Both are covered below.

## Before you start: find out 4 things

Run these on the login node (or read your cluster docs):

| What | How to find it |
| - | - |
| GPU partition name | `sinfo` and look for a partition with GPUs (often `gpu`) |
| Whether you need an account | `sacctmgr show assoc user=$USER` |
| Module names | `module avail python` and `module avail cuda` |
| Do compute nodes have internet | ask the helpdesk; if not, download models on the login node first |

Then edit the lines marked `<-- EDIT` in `run_vllm.slurm`.

---

## Step 1. Connect (plain SSH or VS Code)

Plain terminal:

```bash
ssh yourusername@hpc.your-university.edu     # use the campus VPN if required
```

VS Code:

1. Install the **Remote - SSH** extension (by Microsoft).
2. `Cmd+Shift+P` -> **Remote-SSH: Add New SSH Host** -> `ssh yourusername@hpc.your-university.edu`
3. `Cmd+Shift+P` -> **Remote-SSH: Connect to Host**. VS Code now runs on the cluster and you
   can browse files and open a terminal there.

---

## Step 2. One time environment setup

Do this once on the login node (building the env needs no GPU):

```bash
module purge
module load python/3.11 cuda/12.1       # adjust names to your cluster

python -m venv $HOME/vllm-env
source $HOME/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm

# keep the model cache off your small home quota (point at scratch if you have it)
export HF_HOME=$HOME/hf_cache
```

If compute nodes have **no internet**, download the model now on the login node so it lands
in the cache:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
```

---

## Step 3. Run it

### Option A: batch job (simplest)

```bash
cd path/to/hpc
sbatch run_vllm.slurm        # submit
squeue --me                  # watch it move from PENDING to RUNNING
tail -f vllm_<jobid>.out     # live output (replace <jobid>)
```

The generated text appears in `vllm_<jobid>.out`.

### Option B: interactive GPU shell

```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
# you are now ON a GPU node:
source $HOME/vllm-env/bin/activate
nvidia-smi                   # confirm a GPU is visible
python vllm_demo.py
```

---

## Step 4. Notebooks in VS Code, running on the GPU

This is the nicest way to work interactively.

1. Get an interactive GPU node and start Jupyter on it:

   ```bash
   srun --partition=gpu --gres=gpu:1 --time=02:00:00 --pty bash
   source $HOME/vllm-env/bin/activate
   pip install jupyterlab
   jupyter lab --no-browser --ip=0.0.0.0 --port=8888
   ```

   Note the node name (from `hostname`) and the `http://...:8888/?token=...` URL it prints.

2. Open a tunnel from your Mac so your laptop can reach that node's port. In a new terminal
   on your Mac:

   ```bash
   ssh -N -L 8888:<gpu-node-name>:8888 yourusername@hpc.your-university.edu
   ```

3. In VS Code, open your `.ipynb`, click **Select Kernel** -> **Existing Jupyter Server**,
   and paste the `http://localhost:8888/?token=...` URL. Cells now run on the GPU node.

---

## SLURM command cheat sheet

| Command | Does |
| - | - |
| `sbatch run_vllm.slurm` | submit a batch job |
| `squeue --me` | list your jobs (ST = R running, PD pending) |
| `scancel <jobid>` | cancel a job |
| `sinfo` | list partitions and node states |
| `srun ... --pty bash` | open an interactive session |
| `scontrol show job <jobid>` | detailed job info |

---

## Troubleshooting

| Symptom | Fix |
| - | - |
| `Bfloat16 is only supported on GPUs with compute capability >= 8.0` | you are on a T4/V100; set `DTYPE = "half"` in `vllm_demo.py` |
| CUDA out of memory | use a smaller model, lower `gpu_memory_utilization`, or reduce `max_model_len` |
| `module: command not found` names wrong | run `module avail` and use the exact names your cluster prints |
| Model download hangs on the compute node | compute nodes have no internet; pre-download on the login node (Step 2) |
| Job stuck in `PD` (pending) | the GPU partition is busy; `squeue` shows the queue, just wait |
| `vllm` install pulls the wrong torch | create a fresh venv and `pip install vllm` first, before anything else |

---

## Scaling up once it works

- Bigger model: change `MODEL` in `vllm_demo.py` (a 7-8B model needs a 24GB+ GPU such as an
  A100).
- Multiple GPUs: add `tensor_parallel_size=N` to `LLM(...)` and request `--gres=gpu:N`.
- Serve an OpenAI-compatible API instead of a script:
  `python -m vllm.entrypoints.openai.api_server --model <MODEL> --dtype auto`
