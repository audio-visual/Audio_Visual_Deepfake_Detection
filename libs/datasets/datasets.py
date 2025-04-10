import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, **kwargs):
   """
       A simple dataset builder
   """
   print(f"[in make_dataset], name:{name}, is_training:{is_training}, split:{split}")
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_inference_dataset(name, is_training, split, sub_index, **kwargs):
   """
       A simple dataset builder
   """
   print(f"[in make_dataset], name:{name}, is_training:{is_training}, split:{split}")
   dataset = datasets[name](is_training, split, sub_index, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
