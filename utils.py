import torch
import os
import torch.distributed as dist

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info("==============> Resuming form '{}'....................".format(config.MODEL.RESUME))
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info("=> loaded successfully '{}', epoch '{}'".format(config.MODEL.RESUME, checkpoint['epoch']))
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy

def save_checkpoint(config, model, optimizer, lr_scheduler, logger, epoch, max_accuracy):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'max_accuracy': max_accuracy,
        'epoch': epoch,
        'config': config
    }
    save_path = os.path.join(config.OUTPUT, 'ckpt_epoch_{}.pth'.format(epoch))
    logger.info('{} saving.......'.format(save_path))
    torch.save(save_state, save_path)
    logger.info('{} saved !'.format(save_path))

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()

    return rt