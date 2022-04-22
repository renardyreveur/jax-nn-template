import random

import jax.numpy as jnp


def default_collate_fn(batch: list):
    try:
        if isinstance(batch[0], tuple) or isinstance(batch[0], list):
            return [jnp.stack(x) for x in zip(*batch)]
        elif isinstance(batch[0], str):
            return batch
        elif isinstance(batch[0], dict):
            return [{key: default_collate_fn([b[key] for b in batch])} for key in batch[0]]
        else:
            return jnp.stack(batch)
    except Exception as e:
        print(e)
        raise ValueError("Default Collate Function encountered something it can't handle..")


class DataIterator:
    def __init__(self, dataset, batch_size, shuffle, collate_fn):
        self._dataset = dataset
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._index = 0
        self._shuffle = shuffle
        self._idxs = list(range(len(self._dataset)))

    def __next__(self):
        if self._index < (len(self._dataset) // self._batch_size):
            indices = random.sample(self._idxs, self._batch_size) if self._shuffle else self._idxs[:self._batch_size]
            batch_data = [self._dataset[i] for i in indices]
            self._idxs = list(set(self._idxs) - set(indices))
            return self._collate_fn(batch_data)
        elif self._index == (len(self._dataset) // self._batch_size):
            batch_data = [self._dataset[i] for i in self._idxs]
            return self._collate_fn(batch_data)
        self._index += 1

        raise StopIteration


class BaseDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn if collate_fn is not None else default_collate_fn

    def __len__(self):
        batches = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size:
            batches += 1
        return batches

    def __iter__(self):
        return DataIterator(self.dataset, self.batch_size, self.collate_fn)
