# Towards Accurate Scene Text Recognition with Semantic Reasoning Networks
## This is an unofficial implementation of SRN model for Pytorch.The model has reached the results in the paper on some datasets, and some datasets have not yet reached. Instead of training PVAM and other modules in stages as described in the paper, we train the model end-to-end. The whole project refers to Paddle-OCR.


## Results
|IIIT5K|ic13_1015|ic03_867|ic15_1811|svt_647|svtp_645|cute80_288|
|----|----|----|----|----|----|----|
|94.6|92.4|94.0|0.80|91.7|83.4|84.0|

## Requirements
Pytorch >=1.2.0


## Training

```shell
bash train.sh 
```
--train_data_dir is the train dataset, download the train dataset from [ASTER](https://github.com/ayumiymk/aster.pytorch)

## eval
```shell
bash test.sh
```
--test_data_dir is the eval dataset path.

## Reference
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)  
[ASTER](https://github.com/ayumiymk/aster.pytorch)






