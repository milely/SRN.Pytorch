import torch.nn as nn
import torch
import torch.nn.functional as F


class PrepareEncoder(nn.Module):
    def __init__(self,src_emb_dim,src_max_len,dropout_rate):
        super(PrepareEncoder,self).__init__()
        self.src_emb_dim = src_emb_dim
        self.src_max_len = src_max_len
        self.dropout_rate = dropout_rate
        self.emb = nn.Embedding(num_embeddings=self.src_max_len,embedding_dim=self.src_emb_dim,sparse=False)
        self.drop_out = nn.Dropout(dropout_rate)
    
    def forward(self,src_word,src_pos):
        src_pos_enc = self.emb(src_pos)
        src_pos_enc = src_pos_enc.detach()
        out = src_word + src_pos_enc
        if self.dropout_rate:
            out = self.drop_out(out)
        return out

class PrepareDecoder(nn.Module):
    def __init__(self,vocab_size,src_emb_dim,src_max_len,dropout_rate=0):
        super(PrepareDecoder,self).__init__()
        self.dropout_rate = dropout_rate
        self.word_emb = nn.Embedding(vocab_size,src_emb_dim)
        self.word_emb.weight.data.normal_(0,src_emb_dim**-0.5)      #初始化word_embedding的权重参数

        self.pos_emb = nn.Embedding(src_max_len,src_emb_dim)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self,src_word,src_pos):
        src_word_emb = self.word_emb(src_word)
        src_pos_emb = self.pos_emb(src_pos)
        src_pos_emb = src_pos_emb.detach()
        out = src_word_emb + src_pos_emb
        if self.dropout_rate:
            out = self.drop_out(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,d_key,d_value,d_model,n_head=1,dropout_rate=0):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.d_key = d_key 
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = nn.Linear(d_model,d_key*n_head,bias=False)
        self.k_fc = nn.Linear(d_model,d_key*n_head,bias= False)
        self.v_fc = nn.Linear(d_model,d_value*n_head,bias= False)
        self.proj_fc = nn.Linear(d_value*n_head,d_model,bias=False)
        self.drop_out = nn.Dropout(p = self.dropout_rate)
    
    def _pre_qkv(self,query,key,value,cache=None):
        if key is None:
            key,value = query,query
            static_kv = False
        else:
            static_kv = True

        q = self.q_fc(query)
        b,l_q,d = q.size()
        q = q.view(b,l_q,self.n_head,self.d_key).contiguous()
        q = q.permute(0,2,1,3).contiguous() #b,n,l,d_key

        k = self.k_fc(key)
        v = self.v_fc(value)
        b,l_k,d = k.size()
        b,l_v,d = v.size()
        k = k.view(b,l_k,self.n_head,self.d_key)
        k = k.permute(0,2,1,3).contiguous() #b,n,l_k,d_key
        v = v.view(b,l_v,self.n_head,self.d_value)
        v = v.permute(0,2,1,3).contiguous() #b,n,l_v,d_value
        return q,k,v


    def forward(self,query,key,value,atten_bias=None,cache=None):
        key = query if key is None else key
        value = key if value is None else value
        q,k,v = self._pre_qkv(query,key,value,cache)
        product = torch.matmul(q,k.permute(0,1,3,2))  #q k^T
        product = product * self.d_model**-0.5
        if atten_bias is not None:
            product += atten_bias
        weights = F.softmax(product,dim=-1)
        
        if self.dropout_rate:
            weights = self.drop_out(weights)
        out = torch.matmul(weights,v)
        out = out.permute(0,2,1,3).contiguous()      #b,l,head,dim
        b,l,h,d = out.size()
        out = out.view(b,l,-1)
        out = self.proj_fc(out)
        return out



class FeadForward(nn.Module):    
    def __init__(self,d_inner,d_model,dropout_rate):
        super(FeadForward,self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(d_model,d_inner)
        self.fc2 = nn.Linear(d_inner,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        if self.dropout_rate:
            hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden

class PreLayer(nn.Module):
    def __init__(self,type_lst,d_model=None,dropout_rate=None):
        super(PreLayer,self).__init__()
        self.type_lst = type_lst
        self.layer_lst = []
        for type in type_lst:
            if type =='layer_norm':
                self.layer_lst.append(nn.LayerNorm(d_model))
            elif type == 'drop_out':
                self.layer_lst.append(nn.Dropout(dropout_rate))
        self.layer_lst = nn.Sequential(*self.layer_lst)

    def forward(self,x,residual=None):
        for i,type in enumerate(self.type_lst):
            x = self.layer_lst[i](x)
        return x  

class Encoder_layer(nn.Module):
    def __init__(self,n_head,d_key,d_value,d_model,d_inner,dropout_rate):
        super(Encoder_layer,self).__init__()
        self.pre1 = PreLayer(['layer_norm'],d_model,dropout_rate)
        self.MHA = MultiHeadAttention(d_key,d_value,d_model,n_head,dropout_rate)
        self.pre2 = PreLayer(['drop_out'],d_model,dropout_rate)
        self.pre3 = PreLayer(['layer_norm'],d_model,dropout_rate)
        self.FFN = FeadForward(d_inner,d_model,dropout_rate)
        self.pre4 = PreLayer(['drop_out'],d_model,dropout_rate)
    
    def forward(self,enc_input,atten_bias):
        out = self.MHA(self.pre1(enc_input),None,None,atten_bias)
        out_atten = self.pre2(out) + enc_input   #residual 
        out = self.FFN(self.pre3(out_atten))
        out = self.pre4(out) + out_atten         #residual
        return out

class TransEncoder(nn.Module):
    def __init__(self,n_layer,n_head,d_key,d_value,d_model,d_inner,dropout_rate,src_max_len):
        super(TransEncoder,self).__init__()
        self.PreEncoder = PrepareEncoder(d_model,src_max_len,dropout_rate)
        self.n_layer = n_layer
        for i in range(n_layer):
            self.__setattr__('sub_trans_%d' % i,Encoder_layer(n_head,d_key,d_value,d_model,d_inner,dropout_rate))
        self.prelayer = PreLayer(['layer_norm'],d_model,dropout_rate)

    def forward(self,enc_inputs):
        conv_features,src_pos,atten_bias = enc_inputs
        enc_input = self.PreEncoder(conv_features,src_pos) #
        for i in range(self.n_layer):
            enc_input = self.__getattr__('sub_trans_%d' % i)(enc_input,atten_bias)
        enc_output = self.prelayer(enc_input)
        return enc_output

class ReasonEncoder(nn.Module):
    def __init__(self,n_layer,n_head,d_key,d_value,d_model,d_inner,vocab_size,dropout_rate,src_max_len):
        super(ReasonEncoder,self).__init__()
        self.PreDecoder = PrepareDecoder(vocab_size,d_model,src_max_len)
        self.n_layer = n_layer
        for i in range(n_layer):
            self.__setattr__('sub_reason_%d' % i,Encoder_layer(n_head,d_key,d_value,d_model,d_inner,dropout_rate))
        self.prelayer = PreLayer(['layer_norm'],d_model,dropout_rate)
    
    def forward(self,enc_inputs):
        src_word,src_pos,atten_bias = enc_inputs 
        enc_input = self.PreDecoder(src_word,src_pos)
        for i in range(self.n_layer):
            enc_input = self.__getattr__('sub_reason_%d' % i)(enc_input,atten_bias)
        enc_output = self.prelayer(enc_input)
        return enc_output