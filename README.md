## MPI: Multi-receptive and Parallel Integration for Salient Object Detection

Code for paper "MPI: Multi-receptive and Parallel Integration for Salient Object Detection", by Han Sun, Jun Cen , Ningzhong Liu, Dong Liang, and Huiyu Zhou.

#### Requirement

- python-3.5
- pytorch-1.4.0  
- torchvision
- numpy
- apex
- cv2

#### Usage

Clone this repo into your workstation
```bash
git clone https://github.com/NuaaCJ/MPI.git
```

##### - training

1. Download the pre-trained model for `resnet50`  [resnet50-19c8e357.pth](https://pan.baidu.com/s/1l9Q7VQ3C5As6KVmFswmbLA ) (passwd: resi)

2. Generate edge maps for the training set, or download the file we provide [DUT_TR_edges](https://pan.baidu.com/s/1aCpnzy21s_GSn7gXKD9dNg) (passwd: edge)

3. Modify  `MPI\train_mpi.py` to change both the dataset path and the file save path to your own real path

3. run `train_mpi.py`
```bash
python3.5 train_mpi.py

```

##### - test

1. Download our trained model [MPI_model](https://pan.baidu.com/s/13-C5WDg23d3TEMX3e5z61w) (passwd: mpim) and put it into folder `MPI\models`

2. Modify the dataset path and file save path in the `MPI\test.py` and `MPI\main_function.m` to your own real paths

3. run `test.py`, then the saliency maps will be generated under the corresponding path, and the evaluation scores for the model on the test dataset will be stored in `result.txt`
```bash
python3.5 test.py
```

#### The result saliency maps

Here are saliency maps of our model on five different datasets (DUTS, ECSSD, DUT-OMRON, HKU-IS, PASCAL-S) [The result saliency maps](https://pan.baidu.com/s/1GZbVybeKPLFk6gzmMy_1uQ) (passed: maps)

#### Acknowledgements

The evaluation codes `(MPI\*.m)` we used are provided by [F3net](https://github.com/weijun88/F3Net)
