from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader

from config import get_parse_args
from rl.model import get_recurrent_gnn, load_model, save_model
from utils.logger import Logger
from utils.random_seed import set_seed
from trains.train_factory import train_factory
from datasets.dataset_factory import dataset_factory


def main(args):
    print('==> Using settings {}'.format(args))

    logger = Logger(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    # args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')
    print('Using device: ', args.device)

    print('==> Loading dataset...')
    dataset = dataset_factory[args.dataset](args.data_dir, args)
    print("Statistics: ")
    data_len = len(dataset)
    print("Size: ", len(dataset))

    print('Splitting the dataset into training and validation sets..')
    training_cutoff = int(data_len * args.trainval_split)
    print('# training circuits: ', training_cutoff)
    print('# validation circuits: ', data_len - training_cutoff)
    train_dataset = dataset[:training_cutoff]
    val_dataset = dataset[training_cutoff:]

    # use PyG dataloader
    if len(args.gpus) > 1:
        train_dataset = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) #, num_workers=args.num_workers)
        val_dataset = DataListLoader(val_dataset, batch_size=args.batch_size, shuffle=False) #, num_workers=args.num_workers)
    else:
        train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        val_dataset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('==> Creating model...')
    model = get_recurrent_gnn(args)
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.load_model != '':
        model, optimizer, start_epoch = load_model(model, args.load_model, optimizer, args.resume, args.lr, args.lr_step)

    Trainer = train_factory[args.arch]
    trainer = Trainer(args, model, optimizer)
    trainer.set_device(args.device, args.gpus)

    if args.val_only:
        log_dict_val, _ = trainer.val(0, val_dataset)
        exit(0)

    print('==> Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        mark = epoch if args.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_dataset)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if args.val_intervals > 0 and epoch % args.val_intervals == 0:
            save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(mark)), 
                    epoch, model, optimizer)
        with torch.no_grad():
            log_dict_val, _ = trainer.val(epoch, val_dataset)
        for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if log_dict_val[args.metric] < best:
            best = log_dict_val[args.metric]
            save_model(os.path.join(args.save_dir, 'model_best.pth'), 
                    epoch, model)
        else:
            save_model(os.path.join(args.save_dir, 'model_last.pth'), 
                    epoch, model, optimizer)
        logger.write('\n')
        if epoch in args.lr_step:
            save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)), 
                    epoch, model, optimizer)
            lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)

    main(args)
