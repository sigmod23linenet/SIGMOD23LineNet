# LineNet

This repository contains the source code of LineNet, used for SIGMOD 2023 reviewing.

## How to run LineNet:

1.Place dataset under ./datasets folder. 

1.1. Datasets can be downloaded from [this link](https://drive.google.com/file/d/1VVLtAqWeAB45ziCqRDwWUb0VYKz_Ymhb/view?usp=sharing).

1.2. Organize ./datasets folder as follow:

```
datasets
-EEG
-AirQualityUCI
-Stocks
-TrafficVolume
```

2. Run the following scripts:

```
make run-aq-semihard
make run-eeg-semihard
make run-stocks-semihard
make run-traffic-semihard
```
