from base import BaseDataLoader


class Dataset:
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return idx


class DataLoader(BaseDataLoader):
    def __init__(self, dataset_args, batch_size, shuffle=True, collate_fn=None):
        dataset = Dataset(**dataset_args)
        super().__init__(dataset, batch_size, shuffle, collate_fn)
