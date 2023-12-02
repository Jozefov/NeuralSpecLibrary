import torch
import torch.nn as nn


class ScaledMSE(nn.Module):
    def __init__(self, mass_scale=1.0,
                 intensity_pow=0.5,
                 pred_eps=1e-9,
                 weight_scheme=None,
                 loss_pow=1,
                 invalid_mass_weight=0.0,
                 **kwargs):
        super(ScaledMSE, self).__init__()

        self.mass_scale = mass_scale
        self.intensity_pow = intensity_pow
        self.pred_eps = pred_eps
        self.weight_scheme = weight_scheme
        self.invalid_mass_weight = invalid_mass_weight
        self.loss_pow = loss_pow
        self.loss = torch.nn.MSELoss(reduce='none')

    def forward(self, pred_spect, true_spect):
        SPECT_N = true_spect.shape[1]

        mw = 1 + torch.arange(SPECT_N).to(true_spect.device) * self.mass_scale
        mw = mw.unsqueeze(0)

        w = 1

        eps = self.pred_eps

        pred_weighted = (pred_spect + eps) ** self.intensity_pow * mw * w
        pred_weighted_norm = torch.sqrt((pred_weighted ** 2).sum(dim=1)).unsqueeze(-1)

        true_weighted = (true_spect + eps) ** self.intensity_pow * mw * w

        true_weighted_norm = torch.sqrt((true_weighted ** 2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm

        l = self.loss(pred_weighted_normed, true_weighted_normed)

        lw = 1 + (true_spect < 0.01).float() * self.invalid_mass_weight

        return ((l * lw) ** self.loss_pow).mean()

class ReshapedMSELoss(nn.Module):
    # used for HOMO_LUMO prediction
    def __init__(self):
        super(ReshapedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Default MSE Loss

    def forward(self, predictions, targets):
        # Reshape predictions to match the targets shape
        # predictions shape expected: [batch_size, 1]
        # targets shape expected: [batch_size]
        if predictions.dim() > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)

        # Compute the MSE Loss
        return self.mse_loss(predictions, targets)

loss_fun_dict = {
    "Huber": torch.nn.HuberLoss(),
    "ScaledMSE": ScaledMSE(),
    "MSE": ReshapedMSELoss()
}
