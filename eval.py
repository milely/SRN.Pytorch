import torch
import numpy as np
import sys
import os
from tools.utils import get_vocabulary,get_data
from tools.config import get_args
from model.srn_model import SRNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Eval():
    def __init__(self,model,metric,use_cuda=True,voc_type='LOWERCASE'):
        super(Eval,self).__init__()
        self.model = model
        self.metric = metric
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.voc = get_vocabulary(voc_type)

    def decode(self,gt,pred):
        gt_str = ''
        pred_str = ''
        for i in range(len(gt)):
            if self.voc[gt[i]]=='EOS':
                break
            gt_str +=self.voc[gt[i]]
        for i in range(len(pred)):
            if self.voc[pred[i]]=='EOS':
                break
            pred_str +=self.voc[pred[i]]
        return gt_str,pred_str


    def acc_metric(self,pred_lst,gt_lst,res_name=''):
        count =0
        correct_count =0
        if res_name!='':
            with open(res_name,'w',encoding='utf-8') as f_w:
                for (pred,gt) in zip(pred_lst,gt_lst):
                    gt_txt,pred_txt = self.decode(gt,pred)
                    if pred_txt==gt_txt:
                        correct_count+=1
                    count+=1
                    f_w.write('pred: '+pred_txt+'\n')
                    f_w.write('gt: '+gt_txt+'\n')
                f_w.write('total_num: '+str(count)+'\n')
                f_w.write('correct_num: '+str(correct_count)+'\n')
                f_w.write('acc_rate: '+str(1.0*correct_count / count))
        else:
            for (pred,gt) in zip(pred_lst,gt_lst):
                gt_txt,pred_txt = self.decode(gt,pred)
                if pred_txt==gt_txt:
                    correct_count+=1
                count+=1
        return 1.0*correct_count / count


    def eval(self,data_loader,res_name=''):
        total_pred = []
        total_gt = []
        res =0
        with torch.no_grad():
            self.model.eval()
            print('sample_num:',len(data_loader))
            for i,inputs in enumerate(data_loader):
                images,labels,lens = inputs
                images = images.to(self.device)
                batch_size = images.size(0)
                labels = labels.to(self.device)
                output_dict,loss_dict = self.model(images,labels)
                pred = output_dict['decoded_out'].view(batch_size,-1).cpu().numpy()   #batch_size,
                gt = labels.cpu().numpy()
                assert pred.shape[0] == gt.shape[0]
                total_pred.extend(pred)
                total_gt.extend(gt)
            total_pred = np.array(total_pred,dtype=object)
            total_gt = np.array(total_gt,dtype=object)
            if self.metric == 'acc':
                res = self.acc_metric(total_pred,total_gt,res_name)
        return res


if __name__ =="__main__":
    args = get_args(sys.argv[1:])
    output_type ={'LOWERCASE':38,'ALLCASES':64,'ALLCASES_SYMBOLS':96}
    ocr_model = SRNModel(args.in_channels,output_type[args.voc_type],args.max_len,args.num_heads,args.pvam_layer,args.gsrm_layer,args.hidden_dims)
    ocr_model.load_state_dict(torch.load(args.reuse_model))
    test_dir = args.test_data_dir[0]

    test_dataset = ['IIIT5K_3000','ic13_1015','ic03_867','ic15_1811','svt_647','svt_p_645','cute80_288']
    res_dir = 'res_dir'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    evaluator = Eval(ocr_model,'acc',True,args.voc_type)
    for item in test_dataset:
        with open('test_res.txt','a',encoding='utf-8') as f_w:
            test_dataset,test_dataloader = get_data(os.path.join(test_dir,item),args.voc_type,args.max_len,args.num_test,args.height,args.width,64,args.workers,is_train= False,keep_ratio = args.keep_ratio)
            res = evaluator.eval(test_dataloader,os.path.join(res_dir,item)+'.txt')
            print(item+':\t'+str(res))
            f_w.write(item+':\t'+str(res)+'\n')



