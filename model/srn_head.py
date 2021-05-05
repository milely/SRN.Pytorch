import torch.nn as nn
import torch
import numpy as np
from .trans_layer import TransEncoder,ReasonEncoder
from collections import OrderedDict
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PVAM(nn.Module):
    def __init__(self,in_channel,max_len,num_heads,num_layers,hidden_dims):
        super(PVAM,self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        t = 256
        self.trans_enc = TransEncoder(num_layers,num_heads,int(hidden_dims / num_heads),
                                    int(hidden_dims / num_heads),d_model=hidden_dims,d_inner= hidden_dims,dropout_rate=0.1,src_max_len=t)
        
        self.encoder2embedding = nn.Linear(in_channel,in_channel)     #对feature做embedding
        self.pos2embedding = nn.Embedding(max_len,in_channel)
        self.score = nn.Linear(in_channel,1,bias=False)
    
    def forward(self,inputs):
        b,c,h,w = inputs.size()
        conv_features = inputs.view(-1,c,h*w)
        conv_features = conv_features.permute(0,2,1).contiguous()
        b,t,c = conv_features.size()

        feature_order = torch.arange(t,dtype=torch.long).to(device)
        feature_order = feature_order.unsqueeze(0)
        feature_order = feature_order.expand(b,-1)  #b,256
        enc_inputs = [conv_features,feature_order,None]
        word_features = self.trans_enc(enc_inputs)

        #
        b,t,c = word_features.size()
        
        encoder_features = self.encoder2embedding(word_features)   #b,256,512
        encoder_features = encoder_features.unsqueeze(1).expand(-1,self.max_len,-1,-1) #b,25,256,512

        reading_order = torch.arange(self.max_len,dtype=torch.long).to(device)
        reading_order = reading_order.unsqueeze(0).expand(b,-1) #b,25

        pos_embed = self.pos2embedding(reading_order)     #b,25,512
        pos_embed = pos_embed.unsqueeze(2).expand(-1,-1,t,-1) # b,25,256,512

        attention_weight = torch.tanh(pos_embed + encoder_features)
        attention_weight = torch.squeeze(self.score(attention_weight),-1)  #b,25,256
        attention_weight = F.softmax(attention_weight,dim=-1)  #b,25,256  
        pvam_features = torch.matmul(attention_weight,word_features)
        return pvam_features


def get_gsrm_mask(max_text_length,num_heads):
    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1,
                                  [num_heads, 1, 1]) * [-1e9]
    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2,
                                  [num_heads, 1, 1]) * [-1e9]
    
    return torch.from_numpy(gsrm_slf_attn_bias1).to(device),torch.from_numpy(gsrm_slf_attn_bias2).to(device)



class GSRM(nn.Module):
    """
    双向推理，前向推理和后向推理
    """
    def __init__(self,in_channel,char_num,max_len,num_heads,num_layers,hidden_dims):
        super(GSRM,self).__init__()
        self.char_num = char_num
        self.max_len = max_len 
        self.num_heads = num_heads
        self.cls_op = nn.Linear(in_channel,self.char_num)
        self.for_encoder = ReasonEncoder(n_layer=num_layers,n_head=num_heads,d_key = int(hidden_dims / num_heads),
                                    d_value = int(hidden_dims / num_heads) ,
                                    d_model = hidden_dims,
                                    d_inner = hidden_dims,
                                    vocab_size = self.char_num,dropout_rate= 0.1,src_max_len=max_len)
        self.back_encoder = ReasonEncoder(n_layer=num_layers,n_head=num_heads,d_key = int(hidden_dims / num_heads),
                                    d_value = int(hidden_dims / num_heads) ,
                                    d_model = hidden_dims,
                                    d_inner = hidden_dims,
                                    vocab_size = self.char_num,dropout_rate= 0.1,src_max_len=max_len)
        
        self.mul = lambda x: torch.matmul(x,self.for_encoder.PreDecoder.word_emb.weight.permute(1,0))
    
    def forward(self,inputs):
        """
        两个bias的尺寸为 b,head,25,25   应该是一个下三角的矩阵遮盖后面的字符，和一个上三角矩阵遮盖前面的字符，分别是两个方向的信息。
        """
        pad_idx = self.char_num-1 
        b,t,c = inputs.size()  #b,25,512
        inputs = inputs.view(-1,c)
        cls_res = self.cls_op(inputs)
        word_ids = F.softmax(cls_res,dim=-1).argmax(-1)
        word_ids = word_ids.view(-1,t,1)
        word1 = F.pad(word_ids,[0,0,1,0,0,0],'constant',value=pad_idx)  #在左边填充了一个1
        word1 = word1[:,:-1,:]                                          #在左边填充了一个1,然后右边去掉最后一个元素，使用前面的推理后面的
        word2 = word_ids
        atten_bias1,atten_bias2 =    get_gsrm_mask(self.max_len,self.num_heads)
        reading_order = torch.arange(self.max_len,dtype=torch.long).to(device)
        reading_order = reading_order.unsqueeze(0).expand(b,-1) #b,25
        enc_inputs_1 = [word1.squeeze(-1),reading_order,atten_bias1]
        enc_inputs_2 = [word2.squeeze(-1),reading_order,atten_bias2]
        gsrm_feature1 = self.for_encoder(enc_inputs_1)
        gsrm_feature2 = self.back_encoder(enc_inputs_2)
        gsrm_feature2 = F.pad(gsrm_feature2,[0,0,0,1,0,0],'constant',value=0.)
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        gsrm_features = gsrm_feature1 + gsrm_feature2
        gsrm_out = self.mul(gsrm_features)    #使用最开始的word_embedding矩阵进行映射
        b,t,c = gsrm_out.size()
        gsrm_out = gsrm_out.view(-1,c).contiguous()
        return gsrm_features,cls_res,gsrm_out    # cls_res代表的是上一个模块也就是pvam模块输出的类别信息，gsrm_out代表的是本模块输出的，没有做softmax操作

    







class VSFD(nn.Module):
    def __init__(self,in_channel=512,pvam_ch=512,char_num=38):
        super(VSFD,self).__init__()
        self.char_num = char_num
        self.fc0 = nn.Linear(in_channel*2,pvam_ch)
        self.fc1 = nn.Linear(pvam_ch,char_num)
    
    def forward(self,pvam_feature,gsrm_feature):
        b,t,c1 = pvam_feature.size()
        b,t,c2 = gsrm_feature.size()
        combine_featurs = torch.cat([pvam_feature,gsrm_feature],dim=-1)
        combine_featurs = combine_featurs.view(-1,c1+c2).contiguous()
        atten = self.fc0(combine_featurs)
        atten = torch.sigmoid(atten)
        atten = atten.view(-1,t,c1)
        combine_featurs = atten*pvam_feature +(1-atten)*gsrm_feature
        combine_featurs = combine_featurs.view(-1,c1).contiguous()
        out = self.fc1(combine_featurs)
        return out

class SRNHead(nn.Module):
    def __init__(self,in_channels,out_channels,max_len,num_heads,pvam_layer,gsrm_layer,hidden_dims,**kwargs):
        super(SRNHead,self).__init__()
        self.char_num = out_channels
        self.max_len = max_len
        self.num_heads = num_heads
        self.pvam_layer = pvam_layer
        self.gsrm_layer = gsrm_layer
        self.hidden_dims = hidden_dims
        self.pvam = PVAM(in_channels,max_len,num_heads,self.pvam_layer,hidden_dims)
        self.gsrm = GSRM(in_channels,self.char_num,max_len,num_heads,self.gsrm_layer,hidden_dims)
        self.vsfd = VSFD(in_channels,char_num=self.char_num)
        self.gsrm.back_encoder.PreDecoder.word_emb = self.gsrm.for_encoder.PreDecoder.word_emb
    
    def forward(self,input):
        """
        input b,c,h,w  b,512,8,32
        """
        pvam_features = self.pvam(input)

        gsrm_features,cls_res,gsrm_out = self.gsrm(pvam_features)

        final_out = self.vsfd(pvam_features,gsrm_features)

        if not self.training:
            final_out = F.softmax(final_out,dim=-1)

        decode_out = final_out.argmax(-1)
        predicts = OrderedDict([
            ('predict',final_out),          #预测输出的结果,vsfd输出的概率分布
            ('pvam_feature',pvam_features), #pvam模块输出的特征图
            ('decoded_out',decode_out),     #预测输出的id值，在推理的时候会用到
            ('pvam_out',cls_res),
            ('gsrm_out',gsrm_out),         #pvam和gsrm预测输出的概率，都没有进行softmax的操作
        ])
        return predicts