# %%
"""
This script is able to run a simple training run using Lightning Fabric FSDP.
It is tested running on multiple machines with multiple gpus.

How to use:
- you'll need a pre-tokenized dataset, using the Llama tokenizer (which you can see
how to set up below) in parquet format with a text column of lists of tokens. I used
the scripts from Rocket to do this, using the huggingface wikitext dataset.
Once you have that, change the value for `tokenized_dataset_path` to wherever you saved that.
- you'll need to download the llama tokenizer. You can use the `hf` cli to do that.
Once downloaded, update the path used to construct the tokenizer below
- See `launch_experiment.sh` for how to run with sbatch. You can also use this in
a notebook to run if you'd like.
"""
import signal
import sys
import time
from types import FrameType
from typing import Callable, List, Literal
import dask.dataframe as dd
from jsonargparse import auto_parser
from pathlib import Path
import os
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
import torch
from torch import optim
from torch.nn import Embedding, Linear
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# %%
"""
A very simple Llama test module. `hidden_size` is configurable as a way to change the size
of model being tested. Parameters grow quadratically with `hidden_size`. Minimum is 32, which
runs fine of the P100s for quick testing.
"""
class LightningLlama(torch.nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        config = LlamaConfig(
            vocab_size=128256,
            hidden_size=hidden_size
        )
        self.model = LlamaForCausalLM(config)

    def training_step(self, batch):
        x, attention_mask, labels = batch

        output = self.model(x, attention_mask=attention_mask, labels=labels)

        loss = output.loss

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# %%
tokenizer = AutoTokenizer.from_pretrained(
    '/home/<netid>/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/'
)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>") # for some reason the hub tokenizer doesn't have the pad token set

# %%
tokenized_dataset_path = Path("/home/<path-to-tokenized-dataset-folder>/train")

class WikitextDataset(Dataset):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        assert os.path.isdir(tokenized_dataset_path)
        data = dd.read_parquet(tokenized_dataset_path / '*.parquet').head(128)
        self.text = data['text']

        self.pad_tok = tokenizer.pad_token_id
        self.bos_tok = tokenizer.bos_token_id
        self.eos_tok = tokenizer.eos_token_id
        self.max_sequence_embeddings = 1024
        self.batch_size = batch_size

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text.iloc[index].tolist()
        return (text, text)
    
    def generate_mask(self, size, lens):
        masked_tensor = torch.ones((len(lens), size))
        for i, l in enumerate(lens):
            masked_tensor[i, l:] = 0
        return masked_tensor

    def pad_to_longest(self, batch):
        src, tgt = zip(*batch)

        src_lens = [len(s) for s in src]
        pad_len = max(src_lens)
        src_mask = self.generate_mask(pad_len, src_lens)
        pad_src = [s + [self.pad_tok] * (pad_len - len(s)) for s in src]

        tgt_lens = [len(s) for s in tgt]
        pad_len = max(tgt_lens)
        pad_tgt = [s + [self.pad_tok] * (pad_len - len(s)) for s in tgt]

        pad_src = torch.tensor(pad_src, dtype=torch.long)
        pad_tgt = torch.tensor(pad_tgt, dtype=torch.long)

        return pad_src, src_mask, pad_tgt



# %%
"""
A timer callback, based on the built in Lightning one. Feel free to just delete this if you don't want to
print the total training time.
"""
class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def on_train_start(self):
        self.start_time = time.monotonic()

    def on_train_batch_end(self):
        if self.start_time is None:
            raise Exception('start_time must be initialized')

        curr_time = time.monotonic()
        self.elapsed += curr_time - self.start_time
        self.start_time = curr_time

timer = Timer()


# %%
def is_notebook():
    try:
        get_ipython() # pyright: ignore
        return True
    except NameError:
        return False


# %%
from lightning.fabric.strategies.fsdp import FSDPStrategy

# not sure if this is the ideal auto_wrap_policy
auto_wrap_policy = { Embedding, LlamaDecoderLayer, Linear }
strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy)
callbacks: list[object] = [timer]

# %%
from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric import Fabric
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
import re
from subprocess import call

"""
A pretty basic trainer class. The core is the `fit` method. For distributed training, you need to call `launch`.
"""
class Trainer():
    def __init__(self, default_root_dir: str, callbacks: List[object] = [], devices: int | None=None, num_nodes: int | None=None, strategy: Strategy | str | None=None, environment: ClusterEnvironment | None=None) -> None:
        devices = devices if devices else 1 
        num_nodes = num_nodes if num_nodes else 1
        strategy_dict: dict[Literal['strategy'], Strategy | str] | None = { 'strategy': strategy } if strategy is not None else {}
        environment_dict: dict[Literal['plugins'], ClusterEnvironment] | None = \
           { 'plugins': environment } if environment is not None else {}
        self.fabric = Fabric(devices=devices, num_nodes=num_nodes, callbacks=callbacks, **strategy_dict, **environment_dict)

        self.checkpoint_file = Path(default_root_dir) / 'checkpoint.ckpt'

        self.module: LightningLlama | None = None
        self.optimizer: optim.Optimizer | None = None

        self.iteration = 0

        self.requeue_signal = getattr(environment, 'requeue_signal', None) \
            if getattr(environment, 'auto_requeue', False) else None

    def fit(self, module: LightningLlama, dataset: WikitextDataset):
        module, optimizer = self.fabric.setup(module, module.configure_optimizers())
        self.module = module
        self.optimizer = optimizer
        
        if os.path.isdir(self.checkpoint_file):
            print('loading checkpoint')
            state = self._get_state()
            self.fabric.load(self.checkpoint_file, state)
            timer.elapsed = state.get('elapsed', 0)
            self.iteration = state.get('iteration', 0)
            print(f'loaded {timer.elapsed} {self.iteration} {state}')
            sys.stdout.flush()
       
        dataloader = DataLoader(dataset, batch_size=dataset.batch_size, collate_fn=dataset.pad_to_longest)
        dataloader = self.fabric.setup_dataloaders(dataloader)
        
        self.fabric.call('on_train_start')
        # train for a single epoch
        for i, batch in enumerate(dataloader):
            # skip to saved iteration
            if i < self.iteration:
                print(f'skipping {i}')
                continue
            optimizer.zero_grad()
            loss = module.training_step(batch)
            self.fabric.backward(loss)
            optimizer.step()

            self.iteration = i

            self.fabric.call('on_train_batch_end')
        
        if self.fabric.is_global_zero:
           print(f'Training time (sec): {timer.elapsed}')

    def _get_state(self):
        return { 'module': self.module, 'optimizer': self.optimizer, 'iteration': self.iteration, 'elapsed': timer.elapsed }

    def launch(self, module: LightningLlama, dataset: WikitextDataset):
        def _launch(_):
            print(self.requeue_signal)
            if self.requeue_signal:
                print('Setting up requeue handler')
                sys.stdout.flush()
                signal.signal(self.requeue_signal, self._requeue_handler)

            return self.fit(module, dataset)
        return self.fabric.launch(_launch)

    def _requeue_handler(self, _signum: int, _stack: FrameType | None):
        self.fabric.save(self.checkpoint_file, self._get_state())
        
        if self.fabric.is_global_zero:
            # find job id
            array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
            if array_job_id is not None:
                array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
                job_id = f"{array_job_id}_{array_task_id}"
            else:
                job_id = os.environ["SLURM_JOB_ID"]

            assert re.match("[0-9_-]+", job_id)
            cmd = ["scontrol", "requeue", job_id]

            # requeue job
            print(f"requeing job {job_id}...")
            try:
                result = call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                result = call(" ".join(cmd), shell=True)

            # print result text
            if result == 0:
                print(f"Requeued SLURM job: {job_id}")
            else:
                print(f"Requeuing SLURM job {job_id} failed with error code {result}")
            print(f'saved {self.iteration} {timer.elapsed}')
            sys.stdout.flush()


# %%
from jsonargparse import Namespace

if is_notebook():
    module = LightningLlama(hidden_size=32)
    dataset = WikitextDataset(batch_size=1)

    trainer = Trainer(callbacks=callbacks, strategy='ddp_notebook', default_root_dir='experiments/notebook')
    trainer.launch(module, dataset)
else:
    trainer_overrides = {
        'callbacks': callbacks,
        'strategy': strategy,
        'environment': SLURMEnvironment(requeue_signal=signal.SIGHUP)
    }
    def cli_main(trainer: Trainer, model: LightningLlama, data: WikitextDataset):
        trainer.launch(model, data)

    parser = auto_parser(cli_main)
    cfg = parser.parse_args()
    cfg.update(Namespace(trainer_overrides), 'trainer.init_args', only_unset=False)
    if hasattr(cfg, 'config'):
       del cfg.config # pyright: ignore
    init = parser.instantiate_classes(cfg)

    cli_main(**init)
# %% [markdown]
#
