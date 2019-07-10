# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.path_cfgs import PATH
# from path_cfgs import PATH

import os, torch, random
import numpy as np
from types import MethodType


class Cfgs(PATH):
    def __init__(self):
        super(Cfgs, self).__init__()

        # Set Devices
        # If use multi-gpu training, set e.g.'0, 1, 2' instead
        self.GPU = '0'

        # Set RNG For CPU And GPUs
        self.SEED = random.randint(0, 99999999)

        # -------------------------
        # ---- Version Control ----
        # -------------------------

        # Define a specific name to start new training
        # self.VERSION = 'Anonymous_' + str(self.SEED)
        self.VERSION = str(self.SEED)

        # Resume training
        self.RESUME = False

        # Used in Resume training and testing
        self.CKPT_VERSION = self.VERSION
        self.CKPT_EPOCH = 0

        # Absolutely checkpoint path, 'CKPT_VERSION' and 'CKPT_EPOCH' will be overridden
        self.CKPT_PATH = None

        # Print loss every step
        self.VERBOSE = True


        # ------------------------------
        # ---- Data Provider Params ----
        # ------------------------------

        # {'train', 'val', 'test'}
        self.RUN_MODE = 'train'

        # Set True to evaluate offline
        self.EVAL_EVERY_EPOCH = True

        # Set True to save the prediction vector (Ensemble)
        self.TEST_SAVE_PRED = False

        # Pre-load the features into memory to increase the I/O speed
        self.PRELOAD = False

        # Define the 'train' 'val' 'test' data split
        # (EVAL_EVERY_EPOCH triggered when set {'train': 'train'})
        self.SPLIT = {
            'train': '',
            'val': 'val',
            'test': 'test',
        }

        # A external method to set train split
        self.TRAIN_SPLIT = 'train+val+vg'

        # Set True to use pretrained word embedding
        # (GloVe: spaCy https://spacy.io/)
        self.USE_GLOVE = True

        # Word embedding matrix size
        # (token size x WORD_EMBED_SIZE)
        self.WORD_EMBED_SIZE = 300

        # Max length of question sentences
        self.MAX_TOKEN = 14

        # Filter the answer by occurrence
        # self.ANS_FREQ = 8

        # Max length of extracted faster-rcnn 2048D features
        # (bottom-up and Top-down: https://github.com/peteanderson80/bottom-up-attention)
        self.IMG_FEAT_PAD_SIZE = 100

        # Faster-rcnn 2048D features
        self.IMG_FEAT_SIZE = 2048

        # Default training batch size: 64
        self.BATCH_SIZE = 64

        # Multi-thread I/O
        self.NUM_WORKERS = 8

        # Use pin memory
        # (Warning: pin memory can accelerate GPU loading but may
        # increase the CPU memory usage when NUM_WORKS is large)
        self.PIN_MEM = True

        # Large model can not training with batch size 64
        # Gradient accumulate can split batch to reduce gpu memory usage
        # (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
        self.GRAD_ACCU_STEPS = 1

        # Set 'external': use external shuffle method to implement training shuffle
        # Set 'internal': use pytorch dataloader default shuffle method
        self.SHUFFLE_MODE = 'external'


        # ------------------------
        # ---- Network Params ----
        # ------------------------

        # Model deeps
        # (Encoder and Decoder will be same deeps)
        self.LAYER = 6

        # Model hidden size
        # (512 as default, bigger will be a sharp increase of gpu memory usage)
        self.HIDDEN_SIZE = 512

        # Multi-head number in MCA layers
        # (Warning: HIDDEN_SIZE should be divided by MULTI_HEAD)
        self.MULTI_HEAD = 8

        # Dropout rate for all dropout layers
        # (dropout can prevent overfitting： [Dropout: a simple way to prevent neural networks from overfitting])
        self.DROPOUT_R = 0.1

        # MLP size in flatten layers
        self.FLAT_MLP_SIZE = 512

        # Flatten the last hidden to vector with {n} attention glimpses
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024


        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------

        # The base learning rate
        self.LR_BASE = 0.0001

        # Learning rate decay ratio
        self.LR_DECAY_R = 0.2

        # Learning rate decay at {x, y, z...} epoch
        self.LR_DECAY_LIST = [10, 12]

        # Max training epoch
        self.MAX_EPOCH = 13

        # Gradient clip
        # (default: -1 means not using)
        self.GRAD_NORM_CLIP = -1

        # Adam optimizer betas and eps
        self.OPT_BETAS = (0.9, 0.98)
        self.OPT_EPS = 1e-9


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


    def proc(self):
        assert self.RUN_MODE in ['train', 'val', 'test']

        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)


        # ------------ Seed setup
        # fix pytorch seed
        torch.manual_seed(self.SEED)
        if self.N_GPU < 2:
            torch.cuda.manual_seed(self.SEED)
        else:
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.SEED)

        # fix random seed
        random.seed(self.SEED)

        if self.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(random.randint(0, 99999999))


        # ------------ Split setup
        self.SPLIT['train'] = self.TRAIN_SPLIT
        if 'val' in self.SPLIT['train'].split('+') or self.RUN_MODE not in ['train']:
            self.EVAL_EVERY_EPOCH = False

        if self.RUN_MODE not in ['test']:
            self.TEST_SAVE_PRED = False


        # ------------ Gradient accumulate setup
        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)

        # Use a small eval batch will reduce gpu memory usage
        self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)


        # ------------ Networks setup
        # FeedForwardNet size in every MCA layer
        self.FF_SIZE = int(self.HIDDEN_SIZE * 4)

        # A pipe line hidden size in attention compute
        assert self.HIDDEN_SIZE % self.MULTI_HEAD == 0
        self.HIDDEN_SIZE_HEAD = int(self.HIDDEN_SIZE / self.MULTI_HEAD)


    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''

#
#
# if __name__ == '__main__':
#     __C = Cfgs()
#     __C.proc()





