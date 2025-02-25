import torch


def MAE(pred, true):
    return torch.abs(pred - true)


def MSE(pred, true):
    return (pred - true) ** 2


def MAPE(pred, true):
    return torch.abs((pred - true) / true)


def SMAPE(pred, true):
    # Avoid division by zero by adding a small constant
    denominator = (torch.abs(true) + torch.abs(pred)) / 2 + 1e-8

    # Calculate the SMAPE
    smape_value = torch.mean(torch.abs(pred - true) / denominator)

    return smape_value


def masked_loss(y_pred, y_true, loss_func):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()  # assign the sample weights of zeros to nonzero-values
    loss = loss_func(y_pred, y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_pred - y_true, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def compute_all_metrics(y_pred, y_true):
    mae = masked_loss(y_pred, y_true, MAE).item()
    rmse = masked_rmse_loss(y_pred, y_true).item()
    smape = masked_loss(y_pred, y_true, SMAPE).item()
    return mae, smape, rmse


if __name__ == '__main__':
    y_pred = torch.rand(24, 32, 35)
    y_true = torch.rand(24, 32, 35)

    mae, smape, rmse = compute_all_metrics(y_pred, y_true)
    print(mae, masked_loss(y_pred, y_true, MAE))
    print(rmse, masked_rmse_loss(y_pred, y_true))
    print(smape, masked_loss(y_pred, y_true, SMAPE))
