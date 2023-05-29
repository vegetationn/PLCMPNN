# PLCMPNN

Reimplementation of the CMPNN model by using dgl and PyTorch Lightning.

The original implementation of CMPNN could be referred at [CMPNN](https://github.com/SY575/CMPNN) and [CMPNN-dgl](https://github.com/jcchan23/SAIL/tree/main/Repeat/CMPNN). Thanks a lot for their working! 

The IJCAI 2020 paper could be referred at [Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/Proceedings/2020/0392.pdf).

**Notes: Since the limitation of dgl, we could not implement the reverse edge in the model. The original calculation could be found in the paper Algorithm 1 line 8, so we delete the minus of reverse edge as the substitution and find it will not cause performance decrease a lot.**

## Dependencies
+ cuda == 11.1
+ dgl-cuda11.1 == 0.7.2
+ numpy == 1.23.2
+ ogb == 1.3.5
+ pandas == 1.4.3
+ python == 3.9.13
+ pytorch-lightning
+ RDKit == 2022.9.3
+ scikit-learn == 1.1.3
+ torch == 1.9.1+cu111
+ tqdm == 4.64.1

## Overview

We report the different models and their corresponding results.

### 1. PLCMPNN

These result could be refered from [AttentiveFP](10.1021/acs.jmedchem.9b00959), [SMILES Transformer](https://arxiv.org/abs/1911.04738) and [GROVER](https://arxiv.org/abs/2007.02835). Thanks a lot for their working! 

| Dataset | BACE  | BBBP  | ClinTox | HIV   | SIDER | Tox21 | ToxCast | MUV   | ESOL | FreeSolv | Lipophilicity | QM7 | QM8 | QM9  |
|:---:    |:---:  |:---:  |:---:    |:---:  |:---:  |:---:  |:---:    |:---:  |:---: |:---:     |:---:          |:---:|:---:|:---: |
|Molecules|1513   |2039   |1478     |41127  |1427   |7831   |8577     |93087  |1128  |642       |4200           |6834 |21786|133885|
|Task     |1 GC   |1 GC   |2 GC     |1 GC   |27 GC  |12 GC  |617 GC   |17 GC  |1 GR  |1 GR      |1 GR           |1 GR |12 GR|12 GR |
|Metrics  |AUC-ROC|AUC-ROC|AUC-ROC  |AUC-ROC|AUC-ROC|AUC-ROC|AUC-ROC  |AUC-PRC|RMSE  |RMSE      |RMSE           |MAE  |MAE  |MAE   |
|GraphConv        |0.854±0.011|0.877±0.036|0.845±0.051|-          |0.593±0.035|0.772±0.041|0.650±0.025|-          |1.068±0.050|2.900±0.135|0.712±0.049|118.9±20.2  |0.021±0.001 |-          |
|AttentiveFP      |0.863±0.015|0.908±0.050|0.933±0.020|0.832±0.021|0.605±0.060|0.807±0.020|0.579±0.001|-          |0.853±0.060|2.030±0.420|0.650±0.030|126.7±4.0   |0.0282±0.001|-          |
|Smile Transformer|0.719±0.023|0.900±0.053|0.963±0.064|0.683      |0.559±0.017|0.706±0.021|-          |0.009      |1.144±0.118|2.246±0.237|1.169±0.031|-           |-           |-          |
|GROVER           |0.894±0.028|0.940±0.019|0.944±0.021|-          |0.658±0.023|0.831±0.025|0.737±0.010|-          |0.831±0.120|1.544±0.397|0.560±0.035|72.6±3.8    |0.0125±0.002|-          |
|MPNN             |0.815±0.044|0.913±0.041|0.879±0.054|-          |0.595±0.030|0.808±0.024|0.691±0.013|-          |1.167±0.430|2.185±0.952|0.672±0.051|113.0±17.2  |0.015±0.002 |-          |
|DMPNN            |0.852±0.053|0.919±0.030|0.897±0.040|0.794±0.016|0.632±0.023|0.826±0.023|0.718±0.011|0.045±0.011|0.980±0.258|2.177±0.914|0.653±0.046|105.8±13.2  |0.0143±0.002|3.451±0.174|
|PLCMPNN          |0.874±0.014|0.962±0.004|0.952±0.013|0.835±0.001|0.631±0.010|0.843±0.001|0.743±0.009|0.045±0.007|0.557±0.032|1.391±0.119|0.693±0.017|60.380±1.338|0.012±0.000 |6.767±0.405|

### 2. PLCMPNN-OGB

These result could be refered from [GCN](https://arxiv.org/abs/1611.07308) and [GIN](https://arxiv.org/abs/1810.00826). Thanks a lot for their working! 

| Dataset | ogbg-molbace | ogbg-molbbbp | ogbg-molclintox | ogbg-molhiv | ogbg-molsider | ogbg-moltox21 | ogbg-moltoxcast | ogbg-molmuv | ogbg-molesol | ogbg-mollipo |
|:---:    |:---:         |:---:         |:---:            |:---:        |:---:          |:---:          |:---:            |:---:        |:---:         |:---:         |
|Molecules|1513          |2039          |1477             |41127        |1427           |7831           |8576             |93087        |1128          |4200          |
|Task     |1 GC          |1 GC          |2 GC             |1 GC         |27 GC          |12 GC          |617 GC           |17 GC        |1 GR          |1 GR          |
|Metrics  |AUC-ROC       |AUC-ROC       |AUC-ROC          |AUC-ROC      |AUC-ROC        |AUC-ROC        |AUC-ROC          |AUC-PRC      |RMSE          |RMSE          |
|GCN        |0.689±0.070|0.678±0.024|0.886±0.021|0.760±0.012|0.598±0.015|0.775±0.009|0.667±0.005|0.110±0.029|1.015±0.025|0.771±0.025|
|GIN        |0.735±0.052|0.697±0.019|0.841±0.038|0.771±0.015|0.576±0.016|0.776±0.006|0.661±0.005|0.098±0.027|0.998±0.025|0.704±0.025|
|PLCMPNN-OGB|0.856±0.007|0.703±0.004|0.924±0.019|0.787±0.008|0.646±0.005|0.757±0.005|0.625±0.016|0.106±0.035|0.853±0.007|0.725±0.008|

*Note:*

*(1) GC=Graph Classification, GR=Graph Regression*

*(2) The FreeSolv dataset may need to more works on tuning the hyper-parameters since there are only 642 molecules in this dataset.*

*(3) All split types follow the ratio of 0.8/0.1/0.1 in train/valid/test.*

## Running

### 1. PLCMPNN

To reproduce all the results, run firstly:

`cd dataloader`

`python dataset.py`

it will generate a pickle file in the `data/preprocess` with the same dataset name, this pickle file contain 4 objects:

+ `smiles_list:` All SMILES strings in the dataset.
+ `mols_dict:` Unique SMILES strings -> RDKit mol object.
+ `graphs_dict:` Unique SMILES strings -> dgl graph object.
+ `labels_dict:` Unique SMILES strings -> label list.

Then run:

`python main.py --gpu <gpu num> --data_name <dataset> --split_type <split>`

+ `<gpu num>` is the number of gpu used by the Trainer.

+ `<dataset>` is the dataset name, we provide 14 datasets that mentioned in the overview, more datasets and their results will be updated.

+ `<split>` is the split type name, we provide `[scaffold, random]` in the code.


Others parameters could be refered in the `main.py`.

After running the code, it will create a folder with the format `<args.data_name>_split_<args.split_type>` in the `./result/` folder. Meanwhile, a folder with the format `plcmpnn_seed_<args.seed>_batch_<args.batch_size>_lr_<args.learning_rate>` in the folder with the format `<args.data_name>_split_<args.split_type>`.

If choose scaffold split or random split, the folder will contain:
```
├── result
│   ├── bace_split_random
│   │   ├── plcmpnn_seed_666_batch_64_lr_0.0001
│   │   │   ├── checkpoints
│   │   │   ├── results.txt
│   │   ├── plcmpnn_seed_666_batch_64_lr_0.0001_logs
│   │   ├── test.pickle
│   │   ├── train.pickle
│   │   └── valid.pickle
│   ├── bace_split_scaffold
│   │   ├── plcmpnn_seed_666_batch_64_lr_0.0001
│   │   │   ├── checkpoints
│   │   │   ├── results.txt
│   │   ├── plcmpnn_seed_666_batch_64_lr_0.0001_logs
│   │   ├── test.pickle
│   │   ├── train.pickle
│   │   └── valid.pickle
```

### 2. PLCMPNN-OGB

To reproduce all the results, run firstly:

`cd dataloader`

Then run:

`python main.py --gpu <gpu num> --data_name <dataset>`

+ `<gpu num>` is the number of gpu used by the Trainer.

+ `<dataset>` is the dataset name, we provide 10 datasets that mentioned in the overview, more datasets and their results will be updated.


Others parameters could be refered in the `main.py`.

After running the code, it will create a folder with the format `<args.data_name>` in the `./result/` folder. Meanwhile, a folder with the format `ogbg_seed_<args.seed>_batch_<args.batch_size>_lr_<args.learning_rate>` in the folder with the format `<args.data_name>`.

The folder will contain:
```
├── result
│   ├── ogbg_molbace
│   │   ├── ogbg_seed_666_batch_64_lr_0.0001
│   │   │   ├── checkpoints
│   │   │   ├── results.txt
│   │   ├── ogbg_seed_666_batch_64_lr_0.0001_logs
│   │   ├── mapping
│   │   ├── processed
│   │   ├── raw
│   │   ├── split
│   │   └── RELEASE_v1.txt
```

*Note:*

*Split type is the defult type used by package ogb, the split type name could be seen in the `./result/<args.data_name>/split/` folder.*

## Citation:

Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{ijcai2020-392,
  title     = {Communicative Representation Learning on Attributed Molecular Graphs},
  author    = {Song, Ying and Zheng, Shuangjia and Niu, Zhangming and Fu, Zhang-hua and Lu, Yutong and Yang, Yuedong},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {2831--2838},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/392},
  url       = {https://doi.org/10.24963/ijcai.2020/392},
}
```
