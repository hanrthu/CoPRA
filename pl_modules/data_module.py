from data.sequence_dataset import CustomSeqCollate
from data.structure_dataset import CustomStructCollate
from data.biolip_dataset import BioLiPStructCollate
from data import DataRegister
import pytorch_lightning as pl
import diskcache
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

def get_dataset(data_args:dict=None):
    register = DataRegister()
    dataset_cls = register[data_args.dataset_type]
    return dataset_cls

def get_collate(dataset_type):
    collate_dict = {'sequence_dataset': CustomSeqCollate,
                    'structure_dataset': CustomStructCollate,
                    'biolip_dataset': BioLiPStructCollate,
                    }
    return collate_dict[dataset_type]

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 df_path='', 
                 col_group='fold_0', 
                 batch_size=32, 
                 num_workers=0, 
                 pin_memory=True, 
                 cache_dir=None, 
                 strategy='separate',
                 dataset_args=None, 
                 **kwargs):
        super().__init__()
        self.df_path = df_path
        self.col_group=col_group
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.cache_dir=cache_dir
        self.strategy=strategy
        self.dataset_args=dataset_args
        # print("Dataset Args:", dataset_args)
        
    def setup(self, stage=None):
        if self.cache_dir is None:
            cache = None
        else:
            print("Using diskcache at {}.".format(self.cache_dir))
            cache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')
        
        df = pd.read_csv(self.df_path)
        df_train = df[df[self.col_group].isin(['train'])]
        df_val = df[df[self.col_group].isin(['val'])]
        df_test = df[df[self.col_group].isin(['test'])]
        dataset_cls = get_dataset(self.dataset_args)
        self.train_dataset = dataset_cls(df_train, **self.dataset_args, diskcache=cache)
        self.val_dataset = dataset_cls(df_val, **self.dataset_args, diskcache=cache)


        if len(df_test) > 0 :
            print(f"Using Test Fold to test the model!")
            self.test_dataset = dataset_cls(df_test, **self.dataset_args, diskcache=cache)
        else:
            print(f"Using Validation Fold {self.col_group} to test the model!")
            self.test_dataset = dataset_cls(df_val, **self.dataset_args, diskcache=cache)
    
    def train_dataloader(self):
        if self.dataset_args.dataset_type != 'graph_dataset':
            collate = get_collate(self.dataset_args.dataset_type)
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate(strategy=self.strategy),
            )
        else:
            return GraphLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )

    def val_dataloader(self):
        if self.dataset_args.dataset_type != 'graph_dataset':
            collate = get_collate(self.dataset_args.dataset_type)
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate(strategy=self.strategy),
            )
        else:
            return GraphLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )


    def test_dataloader(self):
        if self.dataset_args.dataset_type != 'graph_dataset':
            collate = get_collate(self.dataset_args.dataset_type)
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate(strategy=self.strategy),
            )
        else:
            return GraphLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )

