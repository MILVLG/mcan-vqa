# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.load_data import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data


class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')
        self.dataset = DataSet(__C)

        self.dataset_eval = None
        if __C.EVAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(__C_eval)


    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            if self.__C.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):

                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]


                    pred = net(
                        sub_img_feat_iter,
                        sub_ques_ix_iter
                    )

                    loss = loss_fn(pred, sub_ans_iter)
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    loss.backward()
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            # if self.__C.VERBOSE:
            #     logfile = open(
            #         self.__C.LOG_PATH +
            #         'log_run_' + self.__C.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + \
                   'ckpt_' + self.__C.CKPT_VERSION + \
                   '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            pred = net(
                img_feat_iter,
                ques_ix_iter
            )
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)

            # Save the whole prediction vector
            if self.__C.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.__C.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.__C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.VERSION + \
                    '.json'

        else:
            if self.__C.CKPT_PATH is not None:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.__C.TEST_SAVE_PRED:

            if self.__C.CKPT_PATH is not None:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.__C.QUESTION_PATH['val']
            ans_file_path = self.__C.ANSWER_PATH['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            vqaEval.evaluate()

            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+'
                )

            else:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")
            logfile.close()


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')




