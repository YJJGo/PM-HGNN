# EM-HGNN
## Requirements
All experiments are conducted on Nvidia RTX4060 GPU (8GB).

Please install the requirements using the following command. (The python version is 3.8.18)
```setup
pip install -r requirements.txt
```

## Datasets
Please extract the datasets (`DBLP.zip`, `ACM.zip`, `IMDB.zip`, `AMiner.zip`) under the folder `'./data/'`.

## Training
To reproduce the results of EM-HGNN on four widely used datasets, please run following commands. 
All dataset-specific hyperparameters are conveniently stored in the **args.yaml** file.

for **DBLP:**
```
python main.py --dataset DBLP
```
for **ACM:**
```
python main.py --dataset ACM
```
for **IMDB:**
```
python main.py --dataset IMDB
```
for **AMiner:**
```
python main.py --dataset AMiner
```