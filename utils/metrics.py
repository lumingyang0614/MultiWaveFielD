from email import message
import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)

#     return mae, mse, rmse, mape, mspe

def metric(pred, true):
    mae, mse, rmse, mape, mspe,CORR, RSE = [],[],[],[],[],[],[]
    
    for i in range(true.shape[1]):
        mae.append(MAE(pred[:,i,:], true[:,i,:]))
        mse.append(MSE(pred[:,i,:], true[:,i,:]))
        rmse.append(RMSE(pred[:,i,:], true[:,i,:]))
        mape.append(MAPE(pred[:,i,:], true[:,i,:]))
        mspe.append(MSPE(pred[:,i,:], true[:,i,:]))
        CORR.append(MAPE(pred[:,i,:], true[:,i,:]))
        RSE.append(MSPE(pred[:,i,:], true[:,i,:]))
    return mae, mse, rmse, mape, mspe, CORR, RSE

