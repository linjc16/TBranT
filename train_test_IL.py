""" Imitation Learning (IL) training. """

import torch
import torch.nn as nn
from torch.optim import optimizer

from torch.utils.data import DataLoader
import math
import numpy as np

import time
import os

from dataset.hdf5_dataloader import collate_fn_original, collate_fn_transformer, collate_fn_transformer_graph, dataset_h5_graph, dataset_h5_ori

from tqdm import tqdm
from utils import AverageMeter, seed_everything, count_parameters
from arguments import get_args
from train_utils import get_policy, get_dataset, get_optimizer, get_scheduler
import pdb


def _train(args, policy, optimizer, scheduler, criterion, train_loader, val_loader):
    # set the policy into train mode
    policy.train()

    # main training loop
    print('Starting training loop...\n')
    for i in range(epoch_start, args.num_epochs - 1):
        # set the policy into train mode
        print('epoch:', i)
        policy.train()
        start_time = time.time()

        running_loss = 0.0
        losses = AverageMeter()
        accs = AverageMeter()

        tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

        for _, batch in enumerate(tk):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_acc = 0.0
            if policy_name == 'Transformer':
                if args.graph:
                    target, node, mip, grid, padding_mask, tree_batch = batch
                    target, node, mip, grid, padding_mask, tree_batch = target.to(device), node.to(device), mip.to(device), grid.to(device), padding_mask.to(device), tree_batch.to(device)
                    logits, _ = policy(grid, padding_mask, node, mip, tree_batch)
                else:
                    target, node, mip, grid, padding_mask = batch
                    target, node, mip, grid, padding_mask = target.to(device), node.to(device), mip.to(device), grid.to(device), padding_mask.to(device)
                    logits, _ = policy(grid, padding_mask, node, mip)
                
                batch_loss += criterion(logits, target)
                batch_acc += (logits.argmax(1).eq(target)).sum()/target.size(0)
        
            else:
                # TreeGate
                for idx, data_tuple in enumerate(batch):
                    target, node, mip, grid = data_tuple
                    target, node, mip, grid = target.to(device), node.to(device), mip.to(device), grid.to(device)
                    logits = policy(grid, node, mip)
                    logits = logits.transpose(1, 0)
                    batch_loss += criterion(logits, target)
                    batch_acc += (logits.max(dim=-1, keepdims=True).indices == target).item()
            
                batch_loss /= float(len(batch))
                batch_acc /= float(len(batch))
            
            batch_loss.backward()
            #nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5, norm_type=2)
            optimizer.step()

            losses.update(batch_loss.detach(), len(batch))
            if policy_name == 'Transformer':
                accs.update(batch_acc.detach(), len(batch))
                losses_avg_tensor = losses.avg
                accs_avg = accs.avg.item()
            else:
                accs.update(batch_acc, len(batch))
                losses_avg_tensor = losses.avg
                accs_avg = accs.avg
            running_loss += batch_loss.item()
            
            if args.lr_decay_schedule:
                tk.set_postfix(loss=losses_avg_tensor.item(), acc=accs_avg, lr=scheduler.get_last_lr()[0])
            else:
                tk.set_postfix(loss=losses_avg_tensor.item(), acc=accs_avg)

        running_loss /= float(num_train_batches)

        if use_scheduler:
            scheduler.step()

        train_time = time.time() - start_time

        # set the policy into eval mode
        policy.eval()
        eval_start = time.time()

        total_correct = 0
        top_k_correct = dict.fromkeys(args.top_k)
        for k in args.top_k:
            top_k_correct[k] = 0
        val_acc_top_k = dict.fromkeys(args.top_k)
        total_loss = 0.0
        nan_counter = 0

        losses = AverageMeter()
        accs = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                if policy_name == 'Transformer':
                    if args.graph:
                        target, node, mip, grid, padding_mask, tree_batch = batch
                        target, node, mip, grid, padding_mask, tree_batch = target.to(device), node.to(device), mip.to(device), grid.to(device), padding_mask.to(device), tree_batch.to(device)
                        logits, _ = policy(grid, padding_mask, node, mip, tree_batch)
                    else:
                        target, node, mip, grid, padding_mask = batch
                        target, node, mip, grid, padding_mask = target.to(device), node.to(device), mip.to(device), grid.to(device), padding_mask.to(device)
                        logits, _ = policy(grid, padding_mask, node, mip)

                    _loss = criterion(logits, target)
                    losses.update(_loss.detach(), len(batch))
                    _acc = (logits.argmax(1).eq(target)).sum()/target.size(0)
                    accs.update(_acc.detach(), len(batch))
                    for k in args.top_k:
                        for idx in range(logits.size(0)):
                            max_k = min(k, logits[idx, :].size(0))
                            top_k_correct[k] += int(target[idx].item() in logits.topk(max_k, dim=1).indices[idx, :])
                else:
                    for idx, data_tuple in enumerate(batch):
                        target, node, mip, grid = data_tuple
                        target, node, mip, grid = target.to(device), node.to(device), mip.to(device), grid.to(device)
                        logits = policy(grid, node, mip)
                        logits = logits.transpose(1, 0)
                        _loss = criterion(logits, target).item()
                        if math.isnan(_loss):
                            nan_counter += 1
                        else:
                            total_loss += _loss
                            _, predicted = torch.max(logits, 1)
                            total_correct += predicted.eq(target.item()).cpu().item()
                            grid_size = grid.size(0)
                            for k in args.top_k:
                                max_k = min(k, grid_size)  # Accounts for when grid_size is smaller than top_k
                                top_k_correct[k] += int(target.item() in logits.topk(max_k, dim=1).indices)

            if policy_name == 'Transformer':
                val_loss = losses.avg.item()
                val_acc = accs.avg.item()
                for k in args.top_k:
                    val_acc_top_k[k] = top_k_correct[k] / float(val_h5.n_data)
            else:
                if nan_counter < val_h5.n_data:
                    val_loss = total_loss / float(val_h5.n_data - nan_counter)
                    val_acc = total_correct / float(val_h5.n_data - nan_counter)
                    for k in args.top_k:
                        val_acc_top_k[k] = top_k_correct[k] / float(val_h5.n_data - nan_counter)
                else:
                    val_loss = np.nan
                    val_acc = np.nan
                    for k in args.top_k:
                        val_acc_top_k[k] = np.nan

                    print('Model overflow on entire val set, hyperparameter configuration is ill-posed, killing the job.')

                    # save the final checkpoint
                    torch.save(
                        {'state_dict': policy.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'args': args,
                         },
                        os.path.join(args.out_dir, 'final_job_crashed_checkpoint.pth.tar')
                    )

                    # exit
                    exit()

            eval_time = time.time() - eval_start

        print(
            "[Epoch {:d}] Train loss: {:.4f}. Train time: {:.2f}sec. "
            "Val loss: {:.4f}. Val acc: {:.2f}%, Val acc top-{}: {:.2f}%, Val acc top-{}: {:.2f}%. Val time: {:.2f}sec.".format(
                i, running_loss, train_time, val_loss, 100 * val_acc, args.top_k[0],
                100 * val_acc_top_k[args.top_k[0]],
                args.top_k[1], 100 * val_acc_top_k[args.top_k[1]], eval_time))

        # checkpoint
        torch.save(
            {'epoch': i,
             'state_dict': policy.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'args': args,
             },
            os.path.join(args.out_dir, 'checkpoint.pth.tar')
        )

        # create per epoch save
        torch.save(
            {'epoch': i,
             'state_dict': policy.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'args': args,
             },
            os.path.join(args.out_dir, 'epoch_{}_checkpoint.pth.tar'.format(i))
        )


def _test(args, policy, optimizer, scheduler, criterion, test_loader):
    # put the policy into eval/validation mode
    print('\nEvaluating on the test set...\n')
    policy.eval()

    total_correct = 0
    top_k_correct = dict.fromkeys(args.top_k)
    for k in args.top_k:
        top_k_correct[k] = 0
    test_acc_top_k = dict.fromkeys(args.top_k)
    total_loss = 0.0
    nan_counter = 0

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():

        if policy_name == 'Transformer':
            for batch in test_loader:
                if args.graph:
                    target, node, mip, grid, padding_mask, tree_batch = batch
                    target, node, mip, grid, padding_mask, tree_batch = target.to(device), node.to(device), mip.to(device), grid.to(device), padding_mask.to(device), tree_batch.to(device)
                    logits, _ = policy(grid, padding_mask, node, mip, tree_batch)
                else:
                    target, node, mip, grid, padding_mask = batch
                    target, node, mip, grid, padding_mask = target.to(device), node.to(device), mip.to(device), grid.to(device), padding_mask.to(device)
                    logits, _ = policy(grid, padding_mask, node, mip)
                _loss = criterion(logits, target)
                losses.update(_loss.detach(), len(batch))
                _acc = (logits.argmax(1).eq(target)).sum()/target.size(0)
                accs.update(_acc.detach(), len(batch))
                for k in args.top_k:
                    for idx in range(logits.size(0)):
                        max_k = min(k, logits[idx, :].size(0))
                        top_k_correct[k] += int(target[idx].item() in logits.topk(max_k, dim=1).indices[idx, :])
            
            
            test_loss = losses.avg.item()
            test_acc = accs.avg.item()
            for k in args.top_k:
                test_acc_top_k[k] = top_k_correct[k] / float(test_h5.n_data)                

        else:
            for batch in test_loader:
                for idx, data_tuple in enumerate(batch):
                    target, node, mip, grid = data_tuple
                    target, node, mip, grid = target.to(device), node.to(device), mip.to(device), grid.to(device)
                    logits = policy(grid, node, mip)
                    logits = logits.transpose(1, 0)
                    _loss = criterion(logits, target).item()
                    if math.isnan(_loss):
                        nan_counter += 1
                    else:
                        total_loss += _loss
                        _, predicted = torch.max(logits, 1)
                        total_correct += predicted.eq(target.item()).cpu().item()
                        grid_size = grid.size(0)
                        for k in args.top_k:
                            max_k = min(k, grid_size)  # Accounts for when grid_size is smaller than top_k
                            top_k_correct[k] += int(target.item() in logits.topk(max_k, dim=1).indices)
            if nan_counter < test_h5.n_data:
                test_loss = total_loss / float(test_h5.n_data - nan_counter)
                test_acc = total_correct / float(test_h5.n_data - nan_counter)
                for k in args.top_k:
                    test_acc_top_k[k] = top_k_correct[k] / float(test_h5.n_data - nan_counter)
            else:
                test_loss = np.nan
                test_acc = np.nan
                for k in args.top_k:
                    test_acc_top_k[k] = np.nan

    print('Test loss: {:.6f}, Test acc: {:.2f}%, Test acc top-{}: {:.2f}%, Test acc top-{}: {:.2f}%'.format(
        test_loss, 100 * test_acc, args.top_k[0], 100 * test_acc_top_k[args.top_k[0]], args.top_k[1],
        100 * test_acc_top_k[args.top_k[1]]))

    print('\n')
    # save the final checkpoint
    torch.save(
        {'state_dict': policy.state_dict(),
         'optimizer': optimizer.state_dict(),
         'scheduler': scheduler.state_dict(),
         'args': args,
         },
        os.path.join(args.out_dir, 'final_checkpoint.pth.tar')
    )



if __name__ == '__main__':

    args = get_args()
    # print(args)

    # set all the random seeds
    seed_everything(args.seed)

    # setup output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # use gpu or cpu
    if args.use_gpu:
        import torch.backends.cudnn as cudnn
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # if final checkpoint exists exit the script
    chkpnt_path = os.path.join(args.out_dir, 'final_checkpoint.pth.tar')
    if os.path.isfile(chkpnt_path):
        print('Final checkpoint exists, experiment has already been run, exiting...')
        exit()
    elif os.path.isfile(os.path.join(args.out_dir, 'final_job_crashed_checkpoint.pth.tar')):
        print('Experiment previously crashed, exiting...')
        exit()

    # load a checkpoint path
    chkpnt_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')
    if os.path.isfile(chkpnt_path):
        chkpnt = torch.load(chkpnt_path)
        epoch_start = chkpnt['epoch']
        print('Checkpoint loaded from path {}, starting at epoch {}...'.format(chkpnt_path, epoch_start))
    else:
        chkpnt = None
        epoch_start = 0

    policy, policy_name = get_policy(args)

    print('Params: {}'.format(count_parameters(policy)))

    policy = policy.to(device)


    if policy_name == 'Transformer':
        if args.graph:
            collate_fn = collate_fn_transformer_graph
        else:
            collate_fn = collate_fn_transformer
    else:
        collate_fn = collate_fn_original
    
    # setup a data loader
    if args.graph:
        dataset_h5 = dataset_h5_graph
    else:
        dataset_h5 = dataset_h5_ori
    
    train_h5, val_h5, test_h5 = get_dataset(args, dataset_h5)

    num_train_batches = train_h5.__len__() // args.train_batchsize
    num_val_batches = val_h5.__len__() // args.eval_batchsize
    num_test_batches = test_h5.__len__() // args.eval_batchsize

    train_loader = DataLoader(dataset=train_h5, batch_size=args.train_batchsize, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_h5, batch_size=args.eval_batchsize, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_h5, batch_size=args.eval_batchsize, shuffle=False, collate_fn=collate_fn)

    optimizer = get_optimizer(args, policy)
    scheduler, use_scheduler = get_scheduler(args, optimizer)

    # setup the loss
    criterion = nn.CrossEntropyLoss().to(device)

    # if checkpoint available, load the policy's and the optimizers parameters
    if chkpnt:
        policy.load_state_dict(chkpnt['state_dict'])
        optimizer.load_state_dict(chkpnt['optimizer'])
        scheduler.load_state_dict(chkpnt['scheduler'])

    _train(args, policy, optimizer, scheduler, criterion, train_loader, val_loader)
    _test(args, policy, optimizer, scheduler, criterion, test_loader)
