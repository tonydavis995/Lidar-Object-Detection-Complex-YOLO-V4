
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import  to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from evaluate import evaluate_mAP
from torch.utils.tensorboard import SummaryWriter
from tfrecord.torch.dataset import TFRecordDataset

import time
import os
import sys


import torch
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1

        batch_size = imgs.size(0)

        targets = targets.to(configs.device, non_blocking=True)
        imgs = imgs.to(configs.device, non_blocking=True)
        total_loss, outputs = model(imgs, targets)

        # For torch.nn.DataParallel case
        if configs.gpu_idx is None:
            total_loss = torch.mean(total_loss)

        # compute gradient and perform backpropagation
        total_loss.backward()
        if global_step % configs.subdivisions == 0:
            optimizer.step()
            # Adjust learning rate
            if configs.step_lr_in_epoch:
                lr_scheduler.step()
                if tb_writer is not None:
                    tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
            # zero the parameter gradients
            optimizer.zero_grad()

        reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        if tb_writer is not None:
            if (global_step % configs.tensorboard_freq) == 0:
                tensorboard_log = get_tensorboard_log(model)
                tb_writer.add_scalar('avg_loss', losses.avg, global_step)
                for layer_name, layer_dict in tensorboard_log.items():
                    tb_writer.add_scalars(layer_name, layer_dict, global_step)

        # Log message
        if logger is not None:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()


def main():
    configs = parse_train_configs()
    # Provision for reproduce results add multi thread distributed processing
    train(configs)


def train(configs):
    configs.gpu_idx = configs.gpu_idx if configs.gpu_idx is None else 0
    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx) if torch.cuda.is_available() else 'cpu')
    configs.subdivisions = int(64 / configs.batch_size)
    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None

    # Import Model Config Yolo V4 / Tiny YoloV4 based on MSE/GIoU/CIoU
    model = create_model(configs)

    # Load weight from a checkpoint
    if configs.pretrained_path is not None:
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint at '{}'".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path))
        if logger is not None:
            logger.info('loaded pretrained model at {}').format(configs.pretrained_path)

    # Resume weights of model from checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at {}".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # Assign model to cuda.
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    model = model.cuda(configs.gpu_idx)

    # Make sure to create optimizer after moving the model to cuda
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")

    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)

    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))
        print('mAP {}'.format(AP.mean()))
        return

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        # train for one epoch
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer)
        if not configs.no_val:
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, logger)
            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

        # Save checkpoint
        if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
