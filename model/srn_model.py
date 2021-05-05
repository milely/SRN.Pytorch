import torch.nn as nn
import torch
from .srn_head import SRNHead
from loss.CE_loss import SRNLoss
from backbone.resnet_fpn import ResNet_FPN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SRNModel(nn.Module):
    def __init__(self,in_channels,out_channels,max_len,num_heads,pvam_layer,gsrm_layer,hidden_dims):
        super(SRNModel,self).__init__()
        self.backbone  = ResNet_FPN(3,50).to(device)
        self.srn = SRNHead(in_channels,out_channels,max_len,num_heads,pvam_layer,gsrm_layer,hidden_dims).to(device)
        self.loss = SRNLoss().to(device)
    def forward(self,x,target_label):
        """
        x: b,c,h,w
        target_label: b,n_class
        """
        encoder_features = self.backbone(x)
        output_dict = self.srn(encoder_features)
        loss_dict = None
        if self.training:
            loss_dict = self.loss(output_dict,target_label)
        return output_dict,loss_dict