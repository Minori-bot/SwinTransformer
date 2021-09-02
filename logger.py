import logging
import sys
import os


def create_logger(output_dir, dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger