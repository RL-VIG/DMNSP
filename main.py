import os
import json
import random

import hydra
import logging
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import statistics
from continuum.metrics import Logger
from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.datasets import build_cl_scenarios
from torch.utils.data import DataLoader, DistributedSampler

WORLD_NUM = 1
@hydra.main(config_path=None, config_name=None, version_base="1.1")
def continual_clip(cfg: DictConfig) -> None:

    set_seed(RANDOM_SEED)

    cfg.workdir = "/***/DMNSP/cil"
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    origin_flag = False
    devices = [0]
    model = load_model(cfg, devices[0], origin_flag)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    print(eval_dataset, eval_dataset)
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names

    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    with open(cfg.log_path, 'w+') as f:
        pass

    acc_list = []
    forgetting_list = []
    metric_logger = Logger(list_subsets=["test"])
    world = WORLD_NUM

    for task_id, _ in enumerate(eval_dataset):

        logging.info(f"Evaluation for task {task_id} has started.")

        model.module.adaptation(task_id, cfg, train_dataset, train_classes_names, world)  # task id 已经传入mode
        eval_sampler = DistributedSampler(eval_dataset[:task_id + 1], num_replicas=world, rank=0)
        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=64, sampler=eval_sampler, num_workers=8)

        for inputs, targets, task_ids in tqdm(eval_loader):
            inputs, targets = inputs.cuda(device=2), targets.cuda(device=2)
            outputs = model.module.cuda(2)(inputs.cuda(2), task_ids)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")


        acc_list.append(100 * metric_logger.accuracy)
        forgetting_list.append(100 * metric_logger.forgetting)

        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last_Cifar100': round(acc_list[-1], 2),
            'avg_Cifar100': round(statistics.mean(acc_list), 2),
            'avg_forgetting': round(statistics.mean(forgetting_list), 2)
        }) + '\n')


RANDOM_SEED = 32

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    continual_clip()
