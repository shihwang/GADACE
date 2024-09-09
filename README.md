# GADACE: Graph Anomaly Detection Combining Attribute Contrast and Structure Reconstruction
This is the official implementation of the submission "GADACE: Graph Anomaly Detection Combining Attribute Contrast and Structure Reconstruction".

### 1. Dependencies (with python = 3.8):
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c dglteam/label/cu116 dgl
conda install pandas
conda install scikit-learn
pip install torch-geometric==2.5.3
pip install pygod
```

### 2. Datasets

|  Dataset | #Nodes | #Edges | Avg. Degree | #Anomalies | Ratio |   Type   |
|:--------:|:------:|:------:|:-----------:|:----------:|:-----:|:--------:|
|   Cora   |  2708  |  5429  |     2.0     |     150    |  5.5% | injected |
| Citeseer |  3327  |  4732  |     1.4     |     150    |  4.5% | injected |
|  Amazon  |  13752 | 515042 |     37.2    |     700    |  5.0% | injected |
|    ACM   |  16484 |  71890 |     4.4     |     600    |  3.6% | injected |
|  Disney  |   124  |   335  |     2.7     |      6     |  4.8% |  organic |
|   Books  |  1418  |  3695  |     2.6     |     28     |  2.0% |  organic |
|  Reddit  |  10984 | 168016 |     15.3    |     366    |  3.3% |  organic |

### 3. Preferred hyperparameters setting

|                                          | Cora | Citeseer | Amazon |  ACM | Disney | Books | Reddit |
|:----------------------------------------:|:----:|:--------:|:------:|:----:|:------:|:-----:|:------:|
| l: weight of one-order neighbor          | 1.0  | 1.0      | 1.0    | 1.0  | 1.0    | 0.6   | 0.7    |
| alpha: weight of original view           | 0.7  | 0.7      | 0.9    | 0.9  | 0.9    | 0.6   | 0.6    |
| beta: weight of context contrast         | 0.7  | 0.7      | 0.9    | 0.9  | 0.9    | 0.7   | 0.7    |
| gamma: weight of sup loss                | 0.3  | 0.9      | 0.2    | 0.1  | 0.4    | 0.5   | 0.4    |
| lamd: lambda of diffusion                | 0.1  | 0.1      | 0.9    | 0.15 | 0.2    | 0.7   | 0.9    |
| t: iteration number of diffusion         | 6    | 5        | 1      | 7    | 12     | 4     | 20     |
| p: weight of diffusion agumented         | 0.7  | 0.4      | 0.5    | 0.7  | 0.5    | 0.7   | 0.9    |
| q: weight of local anomaly score         | 0.7  | 0.7      | 0.7    | 0.7  | 0.9    | 0.9   | 0.3    |
| local-lr: local learning rate            | 1e-3 | 1e-3     | 5e-3   | 5e-3 | 1e-3   | 1e-2  | 1e-3   |
| local-epochs: local training epochs      | 100  | 80       | 80     | 80   | 80     | 80    | 100    |
| structure-lr: global learning rate       | 1e-5 | 1e-5     | 1e-4   | 1e-5 | 1e-5   | 1e-5  | 1e-4   |
| structure-epochs: global training epochs | 200  | 100      | 80     | 120  | 50     | 110   | 100    |
| out-dim: output dimensions               | 512  | 512      | 512    | 512  | 128    | 128   | 128    |


### 4. Anomaly detection
Run `python run.py --data books --l 0.6 --alpha 0.6 --beta 0.7 --gamma 0.5 --lamd 0.7 --t 4 --p 0.7 --q 0.9 --local-lr 1e-2 --local-epochs 80 --structure-lr 1e-5 --structure-epochs 110 --out-dim 128` to perform anomaly detection.
