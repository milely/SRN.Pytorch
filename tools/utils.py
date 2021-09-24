# -*- encoding= utf-8 -*-
import bisect
import random
import string
import warnings
import cv2
import lmdb
import numpy as np
import six
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, sampler
from torchvision import transforms
from .image_aug import warp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING'):
  '''
  voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
  '''
  voc = None
  types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
  if voc_type == 'LOWERCASE':
    voc = list(string.digits + string.ascii_lowercase)
  elif voc_type == 'ALLCASES':
    voc = list(string.digits + string.ascii_letters)
  elif voc_type == 'ALLCASES_SYMBOLS':
    voc = list(string.printable[:-6])
  else:
    raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

  # update the voc with specifical chars
  voc.append(EOS)
  voc.append(PADDING)
  return voc



class LmdbDataset(data.Dataset):
  def __init__(self, root, voc_type, max_len, num_samples,is_train=True,transform=None):
    super(LmdbDataset, self).__init__()
    
    self.env = lmdb.open(root, max_readers=32, readonly=True)

    assert self.env is not None, "cannot create lmdb from %s" % root
    self.txn = self.env.begin()

    self.voc_type = voc_type
    self.transform = transform
    self.max_len = max_len
    self.is_train = is_train
    self.nSamples = int(self.txn.get(b"num-samples"))
    self.nSamples = min(self.nSamples, num_samples)
    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING)
    # print(len(self.voc))
    # print(self.voc)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)
    self.lowercase = (voc_type == 'LOWERCASE')

  def __len__(self):

    return self.nSamples

  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      image = Image.open(buf).convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]

    # reconition labels
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.lowercase:
      word = word.lower()
    ## fill with the padding token
    label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)   #使用padding进行填充
    label_list = []
    for char in word:
      if char in self.char2id:
        label_list.append(self.char2id[char])
      else:
        pass
    ## add a stop token
    label_list = label_list + [self.char2id[self.EOS]]
    if len(label_list) > self.max_len:
      return self[index + 1]
    label[:len(label_list)] = np.array(label_list)

    if len(label) <= 0 :   #空的自动跳到下一个
      return self[index + 1]

    # label length
    label_len = len(label_list)

    if self.transform is not None:
      image = self.transform(image)
    if self.is_train:    #data augmentation
      try:
        img = np.array(image)
        img = warp(img,10)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
      except Exception as e:
        image = image
    return image, label, label_len


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img


class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())


class AlignCollate(object):

  def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    self.imgH = imgH
    self.imgW = imgW
    self.keep_ratio = keep_ratio
    self.min_ratio = min_ratio

  def __call__(self, batch):
    images, labels, lengths = zip(*batch)
    b_lengths = torch.IntTensor(lengths)
    b_labels = torch.LongTensor(labels)

    imgH = self.imgH
    imgW = self.imgW
    if self.keep_ratio:
      ratios = []
      for image in images:
        w, h = image.size
        ratios.append(w / float(h))
      ratios.sort()
      max_ratio = ratios[-1]
      imgW = int(np.floor(max_ratio * imgH))
      imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
      imgW = min(imgW, 400)

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)

    return b_images, b_labels, b_lengths

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.max_len = max([_dataset.max_len for _dataset in self.datasets])
        for _dataset in self.datasets:
            _dataset.max_len = self.max_len

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



def get_dataset(data_dir, voc_type, max_len, num_samples):
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len, num_samples))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples)
  print('total image: ', len(dataset))
  return dataset


def get_dataloader(synthetic_dataset, real_dataset, height, width, batch_size, workers,
                   is_train, keep_ratio):
  num_synthetic_dataset = len(synthetic_dataset)
  num_real_dataset = len(real_dataset)

  synthetic_indices = list(np.random.permutation(num_synthetic_dataset))
  synthetic_indices = synthetic_indices[num_real_dataset:]
  real_indices = list(np.random.permutation(num_real_dataset) + num_synthetic_dataset)
  concated_indices = synthetic_indices + real_indices
  assert len(concated_indices) == num_synthetic_dataset

  sampler = SubsetRandomSampler(concated_indices)
  concated_dataset = ConcatDataset([synthetic_dataset, real_dataset])
  print('total image: ', len(concated_dataset))

  data_loader = DataLoader(concated_dataset, batch_size=batch_size, num_workers=workers,
    shuffle=False, pin_memory=True, drop_last=True, sampler=sampler,
    collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  return concated_dataset, data_loader


def get_data(data_dir, voc_type, max_len, num_samples, height, width, batch_size, workers, is_train, keep_ratio):
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len,num_samples,is_train))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples,is_train)
  print('total image: ', len(dataset))

  if is_train:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=True, pin_memory=True, drop_last=True,
      collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  else:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=False, pin_memory=True, drop_last=False,
      collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  return dataset, data_loader


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer,adjust_lr=1e-5):
    # item_num = global_step // decay_steps
    # n_lr = max(init_lr*math.pow(decay_rate,item_num),min_lr)
    n_lr = adjust_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = n_lr
    print(f'adjust learning rate to {n_lr}')



def test():
  lmdb_path = "../../Data/IIIT5K_3000/"
  train_dataset = LmdbDataset(root=lmdb_path, voc_type='LOWERCASE', max_len=25,num_samples=20)
  batch_size = 1
  train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=AlignCollate(imgH=64, imgW=256, keep_ratio=False))

  for i, (images, labels, label_lens) in enumerate(train_dataloader):
    print(images.size())
    print(label_lens)
    print(labels)

if __name__=='__main__':
  #test()
  a = get_vocabulary('ALLCASES_SYMBOLS')
  print(len(a))


                

            

            

            


        














        
