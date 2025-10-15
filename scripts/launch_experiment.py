# %%
import dask.dataframe as dd
from jsonargparse import lazy_instance
import lightning as L
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.strategies import FSDPStrategy
from pathlib import Path
import os
import re
import signal
import sys
import torch
from torch import optim
from torch.nn import Embedding, Linear
from torch.utils.data import DataLoader
import torchinfo
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# %%
class LightningLlama(L.LightningModule):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        config = LlamaConfig(
            vocab_size=128256,
            hidden_size=hidden_size
        )
        self.model = LlamaForCausalLM(config)

    def training_step(self, batch, batch_idx):
        x, attention_mask, labels = batch

        output = self.model(x, attention_mask=attention_mask, labels=labels)

        loss = output.loss

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# %%
from torch.utils.data import get_worker_info
from torch.distributed import get_rank, get_world_size

tokenized_dataset_path = Path("/home/kobyjl/nobackup/autodelete/rocket-demo/tokenized/")


class DataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, path_to_data, pad_tok, bos_tok, eos_tok, max_sequence_embeddings, cap_length
    ):
        assert os.path.isdir(path_to_data), path_to_data
        self.data = dd.read_parquet(path_to_data / "*.parquet").head(cap_length)
        # Get length of data
        self.length = len(self.data)

        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.max_sequence_embeddings = max_sequence_embeddings

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        world_size = get_world_size()
        process_rank = get_rank()

        # Turn into iterator
        data = self.data.iterrows()

        for index, item in enumerate(data):
            if index % (num_workers * world_size) == (process_rank * num_workers + worker_id):
                item = item[1].values[0].tolist()
                length = len(item)
                if length < 2:
                    continue
                if length <= self.max_sequence_embeddings:
                    # item = np.append(item, self.eos_tok)
                    x = item[: length - 1]
                    y_true = item[1:length]
                else:
                    x = item[: self.max_sequence_embeddings]
                    y_true = item[1 : self.max_sequence_embeddings + 1]
                yield (x, y_true)

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
tokenizer = AutoTokenizer.from_pretrained(
    '/home/kobyjl/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/'
)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")


# %%
class DataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.max_sequence_embeddings = 1024
        self.batch_size = batch_size

    def _dataset(self, subpath):
        return DataSet(
            tokenized_dataset_path / subpath,
            pad_tok=self.pad_id,
            bos_tok=self.bos_id,
            eos_tok=self.eos_id,
            max_sequence_embeddings=self.max_sequence_embeddings,
            cap_length=10000
        )

    def setup(self, stage):
        self.train_dataset = self._dataset("train")
        self.val_dataset = self._dataset("validation")
        self.test_dataset = self._dataset("test")

    def _dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.pad_to_longest)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)


# %%
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment # pyright: ignore

timer = None
class TimerWrapper(Timer):
    def __init__(self) -> None:
        super().__init__()
        global timer
        timer = self


# %%
def is_notebook():
    try:
        get_ipython() # pyright: ignore
        return True
    except NameError:
        return False


# %%
auto_wrap_policy = { Embedding, LlamaDecoderLayer, Linear }
trainer_defaults = {
    'max_epochs': 1,
    'accelerator': 'gpu',
    'strategy': FSDPStrategy(auto_wrap_policy=auto_wrap_policy)
                if is_notebook()
                else lazy_instance(FSDPStrategy, auto_wrap_policy=auto_wrap_policy),
    'callbacks': [TimerWrapper()],
}


# %%
if is_notebook():
    module = LightningLlama(8192)
    dm = DataModule(1)
    dm.setup('')

    trainer = Trainer(**trainer_defaults)
    trainer.fit(module, dm)
else:
    if len(sys.argv) > 1 and sys.argv[1] == '--print-size':
        m = re.search(r'--model\.hidden_size=(\d+)', sys.argv[2])
        if not m:
            print('missing --model.hidden_size=')
            exit()
        module = LightningLlama(int(m.group(1)))
        torchinfo.summary(module.model)
        exit()
    
    trainer_defaults = {
        **trainer_defaults,
        'plugins': { 'class_path': 'lightning.pytorch.plugins.environments.SLURMEnvironment', 'init_args': { 'requeue_signal': signal.SIGHUP }}

    }
    LightningCLI(LightningLlama, DataModule, trainer_defaults=trainer_defaults, save_config_kwargs={ 'overwrite': True })
if not timer:
    raise Exception('timer not set')
print(f'Training time (sec): {timer.time_elapsed('train')}')
