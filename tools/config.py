
import argparse
import math
import sys


sys.path.append('./')
parser = argparse.ArgumentParser(description="Softmax loss classification")

#training config
parser.add_argument('--train_data_dir', nargs='+', type=str, metavar='PATH',
                    default=["../../Data/IIIT5K_3000/"])
parser.add_argument('--test_data_dir', nargs='+', type=str, metavar='PATH',
                    default=["../../Data/IIIT5K_3000/"])
parser.add_argument('--height', type=int, default=64,
                    help="input height, default: 256 for resnet*, ""64 for inception")
parser.add_argument('--width', type=int, default=256,
                    help="input width, default: 128 for resnet*, ""256 for inception")    
parser.add_argument('--reuse_model',type=str,default='',help='the restored model dir')
parser.add_argument('--keep_ratio',type=bool,default=False,
                    help='length fixed or lenghth variable.')
parser.add_argument('--voc_type', type=str, default='LOWERCASE',
                    choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--lr',type= float,default=1e-4,help='learning rate')
parser.add_argument('--metric', type=str, default='acc')
parser.add_argument('--num_train', type=int, default=math.inf)  
parser.add_argument('--num_test', type=int, default=math.inf)  
parser.add_argument('--max_len', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoches', type=int, default=6)
parser.add_argument('--workers', type=int, default=2)

#model config
parser.add_argument('--in_channels',type=int,default=512,help='the srn_head input channel is same as the backbone output')
parser.add_argument('--out_channels',type=int,default=38,help='the output logits dimension')
# parser.add_argument('--max_len',type=int,default=25,help='the pvam decode steps')
parser.add_argument('--num_heads',type=int,default=8,help='the Multihead attention head nums')
parser.add_argument('--pvam_layer',type=int,default=2,help='the pvam default layers')
parser.add_argument('--gsrm_layer',type=int,default=4,help='the gsrm default layers')
parser.add_argument('--hidden_dims',type=int,default=512,help='d_model in transformer')

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args


