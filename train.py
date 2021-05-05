import torch
from tools.utils import AverageMeter,adjust_learning_rate,get_data
from tools.config import get_args
from model.srn_model import  SRNModel
from eval import Eval
import torch.optim as optim
import sys
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, model, test_dataloader, evaluator,reuse_model=''):
        super(Trainer).__init__()
        self.model = model
        self.best_acc = 0
        if reuse_model != '' and not reuse_model.endswith('best.pth'):
            self.iters = int(reuse_model.split('_')[2].split('.')[0])  # 在选用断点的数据进行评测时
        else:
            self.iters = 0                                             # 在不加载参数运行模型时
        self.test_dataloader = test_dataloader
        self.evaluator = evaluator
        self.__init_eval__()                                           # 每次初始运行代码时进行准确率的评测

    def __init_eval__(self):
        begin_res = self.evaluator.eval(self.test_dataloader)
        print(f'Current word_acc of eval dataset: ', begin_res)
        if begin_res > self.best_acc:
            self.best_acc = begin_res

    def train(self, train_dataloader, optimizer, epoch, evaluator):
        self.model.train()
        losses = AverageMeter()
        pvam_loss = AverageMeter()
        vsfd_loss = AverageMeter()

        if epoch == 5:
            adjust_learning_rate(optimizer, 1e-5)

        for i, inputs in enumerate(train_dataloader):
            self.model.train()
            self.iters += 1
            images, labels, lens = inputs
            images = images.to(device)
            batch_size = images.size(0)
            labels = labels.to(device)

            output_dict, loss_dict = self.model(images, labels)
            total_loss = loss_dict['loss']
            pred = output_dict['decoded_out'].view(batch_size, -1).cpu().numpy()[0]
            gt = labels.cpu().numpy()[0]
            optimizer.zero_grad()
            total_loss.backward()
            losses.update(total_loss.item(), batch_size)
            pvam_loss.update(loss_dict['pvam_loss'].item(), batch_size)
            vsfd_loss.update(loss_dict['vsfd_loss'].item(), batch_size)
            # if self.grad_clip > 0:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            if self.iters % 100 == 0:
                print('Epoch:[{}][{}/{}]\t'
                      'Loss:{:.3f}({:.3f})\t'
                      'Pvam_Loss:{:.3f}({:.3f})\t'
                      'Vsfd_Loss:{:.3f}({:.3f})\t'.format(epoch, i + 1, len(train_dataloader), losses.val, losses.avg,
                                                          pvam_loss.val, pvam_loss.avg,
                                                          vsfd_loss.val, vsfd_loss.avg))
                gt_str, pred_str = evaluator.decode(gt, pred)
                print('gt:\t', gt_str[0])
                print('pred:\t', pred_str[0])
            if self.iters % 5000 == 0:
                eval_res = evaluator.eval(self.test_dataloader)
                print('Epoch:[{}][{}/{}]\t'
                      'acc:{:.3f}\t'.format(epoch, i + 1, len(train_dataloader), eval_res))
                if eval_res > self.best_acc:
                    self.model.train()
                    torch.save(self.model.state_dict(), './ckpt/SRN_best.pth')
                    self.best_acc = eval_res
                with open('eval_res.txt', 'a', encoding='utf-8') as f_w:
                    f_w.write(str(eval_res) + '\n')

            if self.iters % 50000 == 0:
                torch.save(self.model.state_dict(), os.path.join('./ckpt/', f'SRN_{epoch}_{self.iters}.pth'))

if __name__=='__main__':
    args = get_args(sys.argv[1:])
    output_type ={'LOWERCASE':38,'ALLCASES':64,'ALLCASES_SYMBOLS':96}
    ocr_model = SRNModel(args.in_channels,output_type[args.voc_type],args.max_len,args.num_heads,args.pvam_layer,args.gsrm_layer,args.hidden_dims)
    if args.reuse_model != '':
        ocr_model.load_state_dict(torch.load(args.reuse_model))
    optimizer = optim.Adam(ocr_model.parameters(), lr=args.lr)
    train_dataset, train_dataloader = get_data(args.train_data_dir, args.voc_type, args.max_len, args.num_train,
                                               args.height, args.width, args.batch_size, args.workers, is_train=True,
                                               keep_ratio=args.keep_ratio)
    test_dataset, test_dataloader = get_data(args.test_data_dir, args.voc_type, args.max_len, args.num_test,
                                             args.height, args.width, args.batch_size, args.workers, is_train=False,
                                             keep_ratio=args.keep_ratio)
    evaluator = Eval(ocr_model, 'acc', True,args.voc_type)
    trainer = Trainer(ocr_model, test_dataloader, evaluator, args.reuse_model)
    for epoch in range(args.epoches):
        trainer.train(train_dataloader, optimizer, epoch, evaluator)