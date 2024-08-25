# MultiWaveFielD


## Requirements

To run the code, you need to install `Python(>=3.9.12)` and `PyTorch(>=1.11.0)` at least. The full requirements are specified in `requirements.txt`.

The `pytorch_wavelets` package should be installed following the instructions of [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

## Data

The Electricity, Solar-energy and Traffic datasets can be downloaded from [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).

The Weather,ETTh2 datasets can be downloaded from [Autoformer](https://github.com/thuml/Autoformer).

You should put the `xx.csv` file into the directory with the datasets' name in `dataset` directory.

For example, the proper file structure should be like:
```
dataset
|-- electricity
| |-- electricity.csv
|--solar
| |-- solar.csv
|--ETT-small
| |-- ETTh2.csv
|--traffic
| |-- traffic.csv
|--weather
| |-- weather.csv
```

## Running the code

The running script is `run.sh`, where you can change any arguments which have been declared in `run.py`.

## Contact

If you have any questions, you can raise an issue or send an e-mail to lumingyang0614@gmail.com

## Acknowledgement

Thanks to the following repos for their codes and datasets.

[https://github.com/fbcotter/pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)

[https://github.com/nnzhan/MTGNN](https://github.com/nnzhan/MTGNN)

[https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)

[https://github.com/alanyoungCN/WaveForM](https://github.com/alanyoungCN/WaveForM
