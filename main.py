import os
import torch
import argparse
import torch.distributed as dist
from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path of config file')
    parser.add_argument('--opts', nargs='+', help="Modify config options by adding 'KEY VALUE' pairs ")

    # easy hyper-params config
    parser.add_argument('--batch-size', type=int, metavar='N', help='batch size for single GPU')
    parser.add_argument('--data-path', type=str, metavar='PATH', help='path of dataset')
    parser.add_argument('--resume', type=str, metavar='PATH', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, metavar='N', help='gradient accumulation steps')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(config):

    dataset_train, dataset_valid, data_loader_train, data_loader_valid = build_loader(config)

if __name__ == '__main__':

    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print("RANK and WORLD_SIZE in environ: '{}/{}'".format(rank, world_size))
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
