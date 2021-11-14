import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search_imagenet import Network
from architect import Architect
from tensorboardX import SummaryWriter
from genotypes import init_space

parser = argparse.ArgumentParser("imagenet")
###################### dss ######################
parser.add_argument('--no_norm', action='store_true', default=False, help='not norm')
parser.add_argument('--rdss_prob', type=float, default=0., help='random dynamic search space probability')
parser.add_argument('--space_config', type=str, default='space_config_imagenet', help='which architecture to save')
parser.add_argument('--arch', type=str, help='which architecture to save')
parser.add_argument('--dss_freq', type=int, default=30, help='frequence of changing dynamic search space')
parser.add_argument('--dss_max_ops', type=int, default=28, help='max ops num in each dynamic search space')
parser.add_argument('--confidence', type=float, default=1.44, help='confidence of ucb')
parser.add_argument('--saliency_type', type=str, default='simple', help='simple, all')
###################### end ######################
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='data/imagenet_split', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=0, help='epoch of begining search')
# parser.add_argument('--gpus', type=str, default='7', help='gpus')
# parser.add_argument('--port', type=str, default='23476', help='tcp port')
parser.add_argument('--note', type=str, default='imagenet', help='note for this run')
parser.add_argument('--resume', action='store_true', default=False, help='resume')
parser.add_argument('--resume_path', type=str, default='', help='resume_path')
parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
args = parser.parse_args()

if args.resume:
    args.save = args.resume_path
else:
    args.save = 'logs/search-{}-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

space_config = init_space(args.space_config)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

data_dir = args.data
# data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
# Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large
CLASSES = 1000
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.autograd.set_detect_anomaly(True)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    # dataset_dir = '/cache/'
    # pre.split_dataset(dataset_dir)
    # sys.exit(1)

    if args.resume:
        state = utils.load_checkpoint(os.path.join(args.save, 'checkpoint.pt'))

    # dataset prepare
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # dataset split
    train_data1 = dset.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_data2 = dset.ImageFolder(valdir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    valid_data = dset.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    num_train = len(train_data1)
    num_val = len(train_data2)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)

    model = Network(args.init_channels,
                    CLASSES,
                    args.layers,
                    criterion,
                    args.confidence,
                    args.dss_max_ops,
                    args.saliency_type,
                    not args.no_norm,
                    primitives=(None if space_config is None else space_config['PRIMITIVES']),
                    space_config=(None if space_config is None else space_config))
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.resume:
        model.load_state_dict(state['model'])
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.module.weights_parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(list(model.module.arch_parameters()) + list(model.module.space_parameters()), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=False)
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_data1)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(train_data2)

    test_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=valid_sampler, num_workers=args.workers)
    train_queue = torch.utils.data.DataLoader(train_data1, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler1, num_workers=args.workers)
    valid_queue = torch.utils.data.DataLoader(train_data2, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler2, num_workers=args.workers)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # architect = Architect(model, args)
    start_epoch = 0
    stage_id = 0
    if args.resume:
        start_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if optimizer_a is not None:
            optimizer_a.load_state_dict(state['arch_optimizer'])
        # validation check
        logging.info(model.module.op_saliency()[0])
        logging.info(model.module.op_saliency()[1])
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
    lr = args.learning_rate
    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)

        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            print(optimizer)

        if space_config is not None:
            if space_config['type'] == 'shrink':
                if stage_id < len(space_config['stage']) and epoch == space_config['stage'][stage_id][0]:
                    model.module.shrink_space(space_config['stage'][stage_id][1])
                    stage_id += 1
            elif space_config['type'] == 'expand':
                if stage_id < len(space_config['stage']) and epoch == space_config['stage'][stage_id][0]:
                    if stage_id == 0:
                        assert space_config['stage'][0][0] == 0
                        model.module.set_space(range(0, len(space_config['PRIMITIVES'])), False)
                    start = 0 if stage_id == 0 else space_config['stage'][stage_id - 1][1]
                    end = space_config['stage'][stage_id][1]
                    model.module.set_space(range(start, end), True)
                    stage_id += 1

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr, epoch)
        logging.info('Train_acc %f', train_acc)

        # validation
        if epoch >= args.begin:
            with torch.no_grad():
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                # test_acc, test_obj = infer(test_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
                # logging.info('Test_acc %f', test_acc)

        logging.info(model.module.op_saliency()[0])
        logging.info(model.module.op_saliency()[1])
        logging.info(model.module._opt_steps_normal)
        logging.info(model.module._opt_steps_reduce)

        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))

        utils.save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'arch_optimizer': None if optimizer_a is None else optimizer_a.state_dict()
        }, os.path.join(args.save, 'checkpoint.pt'))

    if args.arch is not None:
        utils.write_to_file('{} = {}'.format(args.arch, model.module.genotype()), 'genotypes.py')


def train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    end = time.time()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        if epoch >= args.begin:
            if step % args.dss_freq == 0:
                model.module.update_dss()
                if torch.rand([]) < args.rdss_prob:
                    model.module.generate_random_dss()
                else:
                    model.module.generate_dss()
        else:
            # if step == 0:
            #     model.module.update_dss()
            #     model.module.generate_dss()
            model.module.generate_random_dss()

        if epoch >= args.begin:
            optimizer.zero_grad()
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            nn.utils.clip_grad_norm_(model.module.space_parameters(), args.grad_clip)
            optimizer_a.step()
        # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        optimizer_a.zero_grad()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('train %03d [data/s: %.5f][batch/s: %.5f][loss: %e][top1: %f][top5: %f]', step, data_time.avg, batch_time.avg, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    end = time.time()
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('valid %03d [data/s: %.5f][batch/s: %.5f][loss: %e][top1: %f][top5: %f]', step, data_time.avg, batch_time.avg, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
