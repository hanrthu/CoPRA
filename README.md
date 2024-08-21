# CoPRA

This is the official dataset for CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction.

## Environment setup
```
mamba env create -f environment.yml
```

## Datasets and weights
On the way...


## Run finetune on PRA310
```
python run finetune --model_config ./config/models/esm2_650M_rinalmo_struct.yml --data_config ./config/datasets/pdbbind_struct.yml --run_config ./config/runs/finetune_struct.yml
```


## Run Pre-training
```
python run finetune pretune --model_config ./config/models/esm2_650M_rinalmo_struct.yml --data_config ./config/datasets/biolip.yml --run_config ./config/runs/pretune_struct.yml
```

