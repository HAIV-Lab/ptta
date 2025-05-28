from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from typing import Optional, List

import math
import torch.nn.functional as F

from .clip_pseudolabels import InstanceSelector, gererate_partialY
from .loss import PLL_loss

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
        self.e_margin = e_margin
        self.d_margin = d_margin

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.batch_counter = 0
        self.judge = False

        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

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
        #     self.PartialY_CFG.CONF_QUANTILE = 90
        #     self.PartialY_CFG.REGULAR_THRESHOLD = 0.7
        # self.PartialY_CFG.CONF_QUANTILE = 60
        # self.PartialY_CFG.REGULAR_THRESHOLD = 0.9
            # CONF_THRESHOLD CONF_QUANTILE REGULAR_THRESHOLD
        self.criterion = PLL_loss(type="cc", PartialY=None)
        print("PartialY_CFG:", self.PartialY_CFG.__dict__)
    
    def forward(self, ids, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(ids, x)
                # self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self, alpha=0.3):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
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
    
    def init_mb(self, val_loader):
        # with open("selected_ids.pkl", "rb") as f:
        #     import pickle
        #     self.select_ids = pickle.load(f)
        #     return
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
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, ids, x):
        outputs = self.model(x)
        # probs = F.softmax(outputs, dim=-1)
        # entropys = softmax_entropy(outputs)
        # PL_labels, mask_idxs = gererate_partialY(
        #     config=self.PartialY_CFG, 
        #     probs=probs, 
        #     output_logits=outputs,
        # )
        # Select ids that are in self.select_ids
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
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        

        # filter_ids_1 = torch.where(entropys < self.e_margin)
        # if self.current_model_probs is not None:
        #     cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        #     filter_ids_3 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
        #     self.current_model_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_3].softmax(1))
        #     entropys = entropys[filter_ids_1][filter_ids_3]
        # else:
        #     self.current_model_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        #     entropys = entropys[filter_ids_1]
        #     filter_ids_3 = None
        # loss = self.criterion(outputs, PL_labels) 
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        # loss += entropys.mul(1 / (torch.exp(entropys.clone().detach() - self.e_margin))).mean(0)
        # loss += softmax_entropy(outputs).mean(0)
        # loss = softmax_entropy(outputs).mean(0)
        self.batch_counter += 1
        return outputs

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

class FeatureBank:
    def __init__(self, queue_size, neighbor=1):
        self.queue_size = queue_size if queue_size > 0 else 64
        self.features = None
        self.probs = None
        self.ptr = 0
        # accurally, we use the "farthest" samples
        self.refine_method = "nearest_neighbors"
        self.dist_type = "cosine"
        self.num_neighbors = neighbor
        self.num_features_stored = 0
        self.image_bank = None

    def reset(self):
        self.features = None
        self.probs = None
        self.ptr = 0
        self.num_features_stored = 0
        self.image_bank = None

    def update(self, x, features, logits):
        probs = F.softmax(logits, dim=1)

        start = self.ptr
        end = start + features.size(0)
        if (self.features is None or self.probs is None) or self.queue_size == 64:
            self.features = torch.zeros(self.queue_size, features.size(1)).cuda()
            self.probs = torch.zeros(self.queue_size, probs.size(1)).cuda()
            self.image_bank = torch.zeros(self.queue_size, x.size(1), x.size(2), x.size(3))
        idxs_replace = torch.arange(start, end).cuda() % self.features.size(0)
        self.features[idxs_replace, :] = features
        self.probs[idxs_replace, :] = probs
        self.image_bank[idxs_replace, :, :, :] = x.cpu()
        self.ptr = end % len(self.features)
        self.num_features_stored += features.size(0)

    def refine_predictions(self, features):
        if self.refine_method == "nearest_neighbors":
            pred_labels, probs, images, grads = self.soft_k_nearest_neighbors(features)
        elif self.refine_method == "hard_nearest_neighbors":
            pred_labels, probs, images = self.hard_k_nearest_neighbors(features)
        elif self.refine_method is None:
            pred_labels = probs.argmax(dim=1)
            images = None
        else:
            raise NotImplementedError(f"{self.refine_method} not implemented.")

        return pred_labels, probs, images, grads

    def soft_k_nearest_neighbors(self, features):
        pred_probs = []
        pred_images = []
        grads = []
        for feats in features.split(64):
            distances = get_distances(feats, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(self.num_neighbors, dim=1, largest=True)
            # gathered_distances = torch.gather(distances, 1, idxs)
            grad = self.features[idxs].mean(1)
            probs = self.probs[idxs, :].mean(1)
            images = self.image_bank[idxs].mean(1)
            # random_indices = torch.randint(0, min(self.num_features_stored, self.features.size(0)), (feats.size(0), self.num_neighbors))
            # probs = self.probs[random_indices, :].mean(1)
            # images = self.image_bank[random_indices].mean(1)
            pred_probs.append(probs)
            pred_images.append(images)
            grads.append(grad)
        pred_probs = torch.cat(pred_probs)
        pred_images = torch.cat(pred_images)
        grads = torch.cat(grads)
        _, pred_labels = pred_probs.max(dim=1)

        return pred_labels, pred_probs, pred_images, grads
    
    def hard_k_nearest_neighbors(self, features):
        pred_probs = []
        pred_labels = []
        pred_images = []
        for feats in features.split(64):
            distances = get_distances(feats, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(self.num_neighbors, dim=1, largest=False)

            topk_probs = self.probs[idxs, :]
            topk_one_hot = F.one_hot(topk_probs.argmax(dim=2), num_classes=self.probs.size(1)).float()

            weights = 1.0 / (torch.gather(distances, 1, idxs) + 1e-12)
            weighted_one_hot = topk_one_hot * weights.unsqueeze(-1)
            sample_pred_prob = weighted_one_hot.sum(dim=1) / weights.sum(dim=1, keepdim=True)
            pred_probs.append(sample_pred_prob)

            sample_pred_label = sample_pred_prob.argmax(dim=1)
            pred_labels.append(sample_pred_label)

            images = self.image_bank[idxs].mean(1)
            pred_images.append(images)

        pred_probs = torch.cat(pred_probs)
        pred_labels = torch.cat(pred_labels)
        pred_images = torch.cat(pred_images)

        return pred_labels, pred_probs, pred_images

    def get_nearest_or_farthest_features(self, features, nearest=True):
        if nearest:
            distances = get_distances(features, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(1, dim=1, largest=False)
            selected_features = self.features[idxs]
            selected_features = selected_features.squeeze(1)
        else:
            distances = get_distances(features, self.features[:self.num_features_stored], self.dist_type)
            _, idxs = distances.topk(self.num_neighbors, dim=1, largest=True)
            selected_features = self.features[idxs]
        return selected_features

def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


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