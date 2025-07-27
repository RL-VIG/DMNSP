from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F
import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_class_ids_per_task, get_class_names
from . import utils
from .dynamic_dataset import DynamicDataset

DEFAULT_THRESHOLD = 0.985
TOP_SELECT = 1
EPOCH_NUM = 4
TOP_K_RATIO = 0.1
LAMBDA_SCALE = 30
LAYER_NUM = 12

class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, origin_flag, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.origin_flag = origin_flag
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)
        self.prev_gradients = None
        self.visual_cur_matrix = {}
        self.visual_U = {}
        self.loss_list = []



    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names, world):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).cuda(device=2)
        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names, world)



    def train(self, task_id, cfg, train_dataset, train_classes_names, world):

        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)

        train_iter = iter(train_loader)
        EPOCH = EPOCH_NUM
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches


        for k, v in self.model.named_parameters():
            if "adapt" not in k:
                v.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters() if "adapt" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adapt" in k
        ]

        print('========trainable params============', params_name)
        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )
        self.model = self.model.cuda(device=2)

        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]
        texts = clip.tokenize(texts).cuda(device=2)

        self.model.train()

        batch_count = 0
        lamda = [[0 for _ in range(LAYER_NUM)] for _ in range(LAYER_NUM)]
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            inputs, targets = inputs.cuda(device=2), targets.cuda(device=2)
            logits_per_image, _ = self.model.cuda(device=2)(inputs, texts.cuda(device=2), 0, is_train=True)  # 分开

            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            self.loss_list.append(loss)
            print('CELoss: {}'.format(loss))
            optimizer.zero_grad()
            loss.backward()

            if task_id != 0:
                if batch_count == 0:
                    for j in range(LAYER_NUM):
                        activation_visual = self.model.visual.transformer.lora_feature[j]
                        activation_visual = torch.bmm(activation_visual.detach().permute(1, 2, 0),
                                                      activation_visual.detach().permute(1, 0, 2)).sum(dim=0)
                        U_visual, S, Vh = torch.linalg.svd(activation_visual, full_matrices=False)
                        U_visual = U_visual[:, :TOP_SELECT]

                        for k in range(LAYER_NUM):
                            v_visual = self.visual_U[k]

                            normalized_vector_visual = U_visual / torch.norm(U_visual)
                            similarities_visual = []
                            for column_visual in v_visual.t():
                                normalized_column_visual = column_visual / torch.norm(column_visual)
                                cos_sim_visual = torch.dot(normalized_vector_visual.squeeze(),
                                                           normalized_column_visual.squeeze())
                                similarities_visual.append(cos_sim_visual)

                            dot_products_visual = torch.mean(
                                torch.topk(torch.stack(similarities_visual), int(len(similarities_visual) * TOP_K_RATIO))[0])
                            lamda[j][k] = torch.exp(-dot_products_visual) * LAMBDA_SCALE

                    batch_count = batch_count + 1
                for name, params in self.model.named_parameters():

                    for i in range(LAYER_NUM):
                        if 'visual' in name and 'adapt' in name and 'down' in name and 'weight' in name:
                            v = self.visual_U[i]
                            v_ = torch.mm(params.grad.data, v)
                            params.grad.data = torch.mm(v_, v.T)* lamda[int(name.split(".")[3])][i]

                        elif 'visual' in name and 'adapt' in name and 'up' in name and 'weight' in name:
                            v = self.visual_U[i]
                            v_ = torch.mm(v.T, params.grad.data)
                            params.grad.data = torch.mm(v, v_)* lamda[int(name.split(".")[3])][i]

            optimizer.step()

        torch.cuda.empty_cache()

        train_loader_ = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=128,
                                  shuffle=True, num_workers=8)
        counts = 0
        models = self.model.cuda(2)
        for inputs, targets, task_ids in tqdm(train_loader_):
            inputs = inputs.cuda(device=2)
            with torch.no_grad():
                outputs = models(inputs, texts.cuda(2), 0, is_train=False)

            for i in range(LAYER_NUM):
                if len(self.visual_cur_matrix) == i:
                    activation = models.visual.transformer.lora_feature[i]
                    activation = torch.bmm(activation.detach().permute(1, 2, 0),
                                           activation.detach().permute(1, 0, 2)).sum(dim=0)
                    self.visual_cur_matrix[i] = activation

                    U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                    self.visual_U[i] = U[:,TOP_SELECT:]

                else:
                    activation = models.visual.transformer.lora_feature[i]
                    activation = torch.bmm(activation.detach().permute(1, 2, 0),
                                           activation.detach().permute(1, 0, 2)).sum(dim=0)

                    U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                    Ui = torch.cat((self.visual_U[i], U1[:, TOP_SELECT:]), dim=1)
                    self.visual_U[i] = Ui

            counts = counts + 1
            if counts == 1:
                break

        torch.cuda.empty_cache()
        self.model.eval()

class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device, origin_flag) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device, origin_flag)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)