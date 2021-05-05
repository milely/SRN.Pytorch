import torch.nn as nn


class SRNLoss(nn.Module):
    def __init__(self,ignore_index=37):
        super(SRNLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predicts, target_label):
        vsfd_pred = predicts['predict']
        pvam_pred = predicts['pvam_out']
        gsrm_pred = predicts['gsrm_out']

        cost_pvam = self.loss_func(pvam_pred.view(-1, pvam_pred.shape[-1]), target_label.contiguous().view(-1))
        cost_gsrm = self.loss_func(gsrm_pred.view(-1, gsrm_pred.shape[-1]), target_label.contiguous().view(-1))
        cost_vsfd = self.loss_func(vsfd_pred.view(-1, vsfd_pred.shape[-1]), target_label.contiguous().view(-1))
        sum_cost = cost_pvam * 3.0 + cost_vsfd + cost_gsrm * 0.5
        return {'loss': sum_cost, 'pvam_loss': cost_pvam, 'vsfd_loss': cost_vsfd}