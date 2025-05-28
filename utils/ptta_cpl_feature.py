from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from typing import Optional, List

import math
import torch.nn.functional as F

from .clip_pseudolabels import InstanceSelector, gererate_partialY
from .loss import PLL_loss
from .ptta_eata_image_feature import FeatureBank

class DictToClass:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class Config:
    def __init__(self):
        self.VIS_ENCODER = "ViT-B/32"
        self.MODALITY = "image"
        self.LEARNING_PARADIGM = "ul"
        self.N_PSEUDOSHOTS = 16
        self.PartialY_CFG = {
            "USE_SOFT_PARTIAL": False,
            "CANDIDATE_METHOD": "CPL",
            "CONF_THRESHOLD": "quantile",
            "CONF_QUANTILE": 60,
            "REGULAR_THRESHOLD": 0.9,
            "TARGET_PARTIAL_RATIO": 0.5,
            "INIT_PARTIAL_RATIO": 0.0,
        }
        self.PartialY_CFG = DictToClass(self.PartialY_CFG)
        self.Selector_CFG = {
            "PSEUDOSHOTS_PERCENT": 0
        }
        self.Selector_CFG = DictToClass(self.Selector_CFG)
        # PROMPT_TEMPLATE: 'imported in main.py'
        self.PROMPT_TEMPLATE = "imported in main.py"
        # VIS_PREFIX_INIT: "normal"
        self.VIS_PREFIX_INIT = "normal"
        # PREFIX_SIZE: 16
        self.PREFIX_SIZE = 16
# 创建配置对象
obj_conf = Config()

class CPL(nn.Module):
    def __init__(self, model, optimizer, e_margin=math.log(1000)*0.40, d_margin=0.05, loss2_weight=3, 
                 queue_size=1000, fisher_alpha=2000, neighbor=1):
        super().__init__()
        self.model = model
        # self.model = ResNet50FeatureExtractor(model)
        
        self.optimizer = optimizer
        self.steps = 1
        self.episodic = False
        self.fishers = None

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        # self.e_margin = e_margin
        a = 0.4
        print("e_margin:", a)
        self.e_margin = math.log(1000)*a
        self.d_margin = d_margin

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.batch_counter = 0
        self.judge = False

        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.memory_bank = FeatureBank(queue_size, neighbor)
        self.alpha = 1 / (neighbor + 1)
        self.loss2_weight = loss2_weight
        self.fisher_alpha = fisher_alpha
        PartialY_CFG = {
            "USE_SOFT_PARTIAL": False,
            "CANDIDATE_METHOD": "CPL",
            "CONF_THRESHOLD": "quantile",
            # "CONF_QUANTILE": 60,
            "CONF_QUANTILE": 90,
            # "REGULAR_THRESHOLD": 0.8,
            "REGULAR_THRESHOLD": 0.7,
            "TARGET_PARTIAL_RATIO": 0.5,
            "INIT_PARTIAL_RATIO": 0.0,
        }
        self.PartialY_CFG = DictToClass(PartialY_CFG)
        imagenet_classes = [f"class_{i}" for i in range(1000)]
        label_to_idx = {c: i for i, c in enumerate(imagenet_classes)}
        self.config = obj_conf
        self.Selector = InstanceSelector(label_to_idx=label_to_idx, cfg=obj_conf)
        # if model.__class__.__name__.lower() == 'visiontransformer':
        #     self.PartialY_CFG.CONF_QUANTILE = 80
        #     self.PartialY_CFG.REGULAR_THRESHOLD = 0.8
        self.criterion = PLL_loss(type="cc", PartialY=None)
        print("PartialY_CFG:", self.PartialY_CFG.__dict__)

    def init_mb(self, val_loader):
        probs_list = []
        ids_list = []
        logits_list = []
        from tqdm import tqdm
        print("computing global filter")
        for ids, x, y in tqdm(val_loader):
            x = x.cuda()
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)
            ids_list.extend(ids)
            logits_list.append(logits)

        probs_list = torch.cat(probs_list, dim=0).float()
        logits_list = torch.cat(logits_list, dim=0).float()

        PL_labels, mask_idxs = gererate_partialY(
            config=self.PartialY_CFG, 
            probs=probs_list, 
            output_logits=logits_list,
        )

        selected_idxs, info_2 = self.Selector.select_topk_for_eachcls(
            PL_labels=(PL_labels > 1e-7).float()[mask_idxs],
            output_all=logits_list[mask_idxs],
            indexs_all=torch.arange(len(ids_list))[mask_idxs],
            K_max=self.config.N_PSEUDOSHOTS,
            candidate_method=self.PartialY_CFG.CANDIDATE_METHOD,
        )
        print(mask_idxs.shape, selected_idxs.shape)
        # print(mask_idxs)
        self.select_ids = torch.tensor(ids_list)[mask_idxs][selected_idxs]
        # self.select_ids = torch.tensor(ids_list)[selected_idxs]
    
    def forward(self, ids, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1 = self.forward_and_adapt_eata(ids, x)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                # self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self, alpha=0.3):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory_bank.reset()
        # if self.model_state is None or self.optimizer_state is None:
        #     raise Exception("cannot reset without saved model/optimizer state")
        # # 假设这里有另一个模型状态字典（比如来自不同阶段训练的模型等情况，示例中简单模拟一个）
        # another_model_state_dict = deepcopy(self.model.state_dict())
        # # 调用函数处理BN层参数进行融合平均等操作
        # new_model_state_dict = filter_and_average_bn_params(another_model_state_dict, self.model_state, self.model, alpha)
        # self.model.load_state_dict(new_model_state_dict, strict=True)
        # self.optimizer.load_state_dict(self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    
    def memory(self, x, y):
        self.memory_bank[y] = x
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_eata(self, ids, x):
        if self.model.__class__.__name__.lower() == 'visiontransformer':
            with torch.no_grad():
                feature = self.model.forward_features(x)
            feature = feature.requires_grad_(True)
            outputs = self.model.head(feature)
        else:
            outputs, feature = self.model(x, return_feature=True)
        loss = softmax_entropy(outputs).mean()
        grads = torch.autograd.grad(loss, feature)[0]
        self.model.zero_grad()
        # PL_labels, mask_idxs = gererate_partialY(
        #     config=self.PartialY_CFG, 
        #     probs=probs, 
        #     output_logits=outputs,
        # )
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        
        filter_ids_1 = torch.where(entropys < self.e_margin)
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_3 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            self.current_model_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_3].softmax(1))
            if self.memory_bank.queue_size > 0:
                if outputs[filter_ids_1][filter_ids_3].size(0) > 0:
                    self.memory_bank.update(x[filter_ids_1][filter_ids_3], grads[filter_ids_1][filter_ids_3].clone().detach(), outputs[filter_ids_1][filter_ids_3].clone().detach())
                elif outputs[filter_ids_1].size(0) > 0:
                    self.memory_bank.update(x[filter_ids_1], grads[filter_ids_1].clone().detach(), outputs[filter_ids_1].clone().detach())
            # self.memory_bank.update(x[filter_ids_1], features[filter_ids_1].clone().detach(), outputs[filter_ids_1].clone().detach())
            entropys = entropys[filter_ids_1][filter_ids_3]
        else:
            self.current_model_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
            # if self.memory_bank.queue_size > 0:
            #     self.memory_bank.update(x[filter_ids_1], grads[filter_ids_1].clone().detach(), outputs[filter_ids_1].clone().detach())
            entropys = entropys[filter_ids_1]
            filter_ids_3 = None
        # self.memory_bank.update(x, grads.clone().detach(), outputs.clone().detach())
            # loss2 = F.kl_div(outputs2.log_softmax(dim=-1), probs, reduction='none').sum(1).mul(1 / (torch.exp(eentropy.clone().detach() - self.e_margin))).mean() * self.loss2_weight
            # loss2 = F.kl_div(outputs2.log_softmax(dim=-1), probs, reduction='none').sum(1).mean() * self.loss2_weight
        # print(entropys.shape[0] / x.size(0))
        # loss = entropys.mean() * 3
        selected_mask = torch.tensor([i in self.select_ids for i in ids], dtype=torch.bool)
        selected_ids = ids[selected_mask]
        # selected_x = x[selected_mask]
        selected_outputs = outputs[selected_mask]
        # selected_probs = probs[selected_mask]
        # selected_entropys = entropys[selected_mask]
        # Update model with selected ids
        if selected_ids.size(0) > 0:
            # loss = self.criterion(selected_outputs, PL_labels[selected_mask])
            loss = softmax_entropy(selected_outputs).mean()
            if self.memory_bank.num_features_stored > 0:
                pred_labels, probs, img, grads_m = self.memory_bank.refine_predictions(grads.clone().detach())
                alpha = self.alpha
                x = x * alpha + img.cuda() * (1 - alpha)
                probs = outputs.softmax(dim=-1) * alpha + probs.cuda() * (1 - alpha)
                outputs2 = self.model(x)
                loss2 = F.kl_div(outputs2.log_softmax(dim=-1), probs, reduction='batchmean') * self.loss2_weight
                # loss += loss2 * (1 / (torch.exp(eentropy.clone().detach() - self.e_margin))).mean()
                loss += loss2
        if selected_ids.size(0) > 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_counter += 1
        return outputs, entropys.size(0), 0

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

@torch.jit.script
def compute_gradient(x):
    x.requires_grad_(True)
    entropy = softmax_entropy(x)
    grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(entropy)])
    grad = torch.autograd.grad(outputs=[entropy], inputs=[x], grad_outputs=grad_outputs, create_graph=False, retain_graph=False)[0]
    return grad

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model



def filter_and_average_bn_params(state_dict_a, state_dict_b, model, alpha=0.3):
    """
    平均两个参数字典中所有BN层的仿射参数，并过滤掉不匹配的键。

    参数:
        state_dict_a (dict): 模型A的状态字典。
        state_dict_b (dict): 模型B的状态字典。
        model (nn.Module): 目标模型，用于确定哪些键是有效的。

    返回:
        dict: 包含平均后BN层参数的新状态字典，其他层保持state_dict_a的参数。
    """
    # 创建一个新的状态字典，初始化为state_dict_a中存在于model.state_dict()中的键值对
    filtered_state_dict = {k: v for k, v in state_dict_a.items() if k in model.state_dict()}

    # 检查并打印被过滤掉的键
    missing_keys = set(state_dict_a.keys()) - set(filtered_state_dict.keys())
    if missing_keys:
        print("The following keys are missing from the current model and will be ignored:")
        for key in missing_keys:
            print(key)

    # 获取目标模型的状态字典作为基础，先复制state_dict_a中对应的部分过来
    target_state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in filtered_state_dict.items()}

    # 定义BN层相关的键（权重和偏置）
    bn_keys = ['weight', 'bias']
    # 遍历目标模型的状态字典，找到BN层相关的键并计算平均值
    for key in model.state_dict().keys():
        if any(bn_key in key for bn_key in bn_keys) and key in state_dict_a and key in state_dict_b:
            with torch.no_grad():
                # 计算加权平均，这里简单地取指定权重（示例中0.3和0.7，可按需调整）
                # target_state_dict[key] = alpha * state_dict_a[key] + (1 - alpha) * state_dict_b[key]
                target_state_dict[key] = alpha * state_dict_b[key] + (1 - alpha) * state_dict_a[key]

    return target_state_dict