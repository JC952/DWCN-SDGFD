import logging
import os
from datetime import datetime
import numpy as np


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(message)s")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
def result_log(Indicators="Accuracy",target="",source="",results=None):
    results = [round(x, 2) for x in results]
    mean = np.mean(results)
    std = np.std(results)
    results.append(f"{mean:.3f}±{std:.3f}")
    logging.info(f"Indicators:{Indicators} Task:{source}->{target} Mean_Var: {results}")
def setup_logging(args, save_dir):
    """Sets up logging configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    setlogger(os.path.join(save_dir, args.model_name + '.log'))
    logging.info("\n")
    time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
    logging.info('{}'.format(time))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))


