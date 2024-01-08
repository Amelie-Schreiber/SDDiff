import os
import time
import yaml
import shutil
import logging
import argparse
from glob import glob
from easydict import EasyDict
from tqdm.auto import tqdm
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader


from datasets import ConformationDataset
from transforms import *

from models.models import DualEncoderEpsNetwork

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir

def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_checkpoint_path(folder, it=None):
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]


    
if __name__ == '__main__':
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]

    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, shuffle=True))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # Model
    logger.info('Building model...')
    model = DualEncoderEpsNetwork(config.model).to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
        betas=(config.train.optimizer.beta1, config.train.optimizer.beta2)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.train.scheduler.factor,
        patience=config.train.scheduler.patience
    )

    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        # optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        # optimizer_global.load_state_dict(ckpt['optimizer_global'])
        # optimizer_local.load_state_dict(ckpt['optimizer_local'])
        # scheduler_global.load_state_dict(ckpt['scheduler_global'])
        # scheduler_local.load_state_dict(ckpt['scheduler_local'])

    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss, loss1, loss2 = model.get_loss(
            atom_type=batch.atom_type,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            it=it
        )
        loss = loss.mean()
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        if not orig_grad_norm.isnan().any():
            optimizer.step()

            logger.info(f'[Train] Iter {it:>07d} | Loss {loss.item():<6.2f}  '+ \
                        f'| Loss1 {loss1.item():<6.2f}  '+\
                        f'| Loss2 {loss2.item():<6.2f}  ')
                        # f'|Loss(pseudo) {loss_pseudo:<6.2f}')
            writer.add_scalar('train/loss', loss, it)
            writer.add_scalar('train/loss1', loss1.item(), it)
            writer.add_scalar('train/loss2', loss2.item(), it)
            # writer.add_scalar('train/loss2_global', loss2_global.mean(), it)
            # writer.add_scalar('train/loss2_local', loss2_local.mean(), it)
            # writer.add_scalar('train/loss_pseudo', loss_pseudo, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            # writer.add_scalar('train/lr_local', optimizer_local.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm, it)
            writer.flush()
        else:
            logger.info(f'[Train] Iter {it:>07d} | warning!!! grad_norm with nan!!! no backward this it')
    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_loss1, sum_n1 = 0, 0
        sum_loss2, sum_n2 = 0, 0
        # sum_loss_pseudo, sum_n_pseudo = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch = batch.to(args.device)
                loss, loss1, loss2 = model.get_loss(
                    atom_type=batch.atom_type,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                )
                sum_loss += loss.sum().item()
                sum_n += 1
                sum_loss1 += loss1.sum().item()
                sum_n1 += 1
                sum_loss2 += loss2.sum().item()
                sum_n2 += 1
                # sum_loss_pseudo += loss_pseudo.sum().item()
                # sum_n_pseudo += 1

        avg_loss = sum_loss / sum_n
        avg_loss1 = sum_loss1 / sum_n1
        avg_loss2 = sum_loss2 / sum_n2
        # avg_loss_pseudo = sum_loss_pseudo / sum_n_pseudo
        
        # scheduler.step(avg_loss)


        logger.info(f'[Validate] Iter {it:>07d} | Loss {loss.item():<6.2f}  '+ \
                    f'| Loss1{avg_loss1:<6.2f}  '+\
                    f'| Loss2{avg_loss2:<6.2f}  ')
                    # f'Loss(pseudo) {avg_loss_pseudo:<6.2f}')
        
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss1', avg_loss1, it)
        writer.add_scalar('val/loss2', avg_loss2, it)
        # writer.add_scalar('val/loss_pseudo', avg_loss_pseudo, it)
        writer.flush()
        return avg_loss

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

