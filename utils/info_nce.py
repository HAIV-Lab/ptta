# info_nce.py
import torch
import torch.nn.functional as F

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='paired'):
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ transpose(positive_key)
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

# Modify the forward_and_adapt_eata method to include info_nce loss

# def forward_and_adapt_eata(self, x, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0, target = None, logging = None):
#     outputs, features = self.model(x)
#     entropys = softmax_entropy(outputs)
#     filter_ids_1 = torch.where(entropys < self.e_margin)
#     filter_ids_2 = torch.where(entropys >= self.e_margin)
#     alpha = 0.2
#     self.memory_bank.update(x[filter_ids_1], features[filter_ids_1].clone().detach(), outputs[filter_ids_1].clone().detach())
#     if self.memory_bank.num_features_stored > 2000:
#         pred_labels, probs, img = self.memory_bank.refine_predictions(features[filter_ids_2].clone().detach())
#         x[filter_ids_2] = x[filter_ids_2] * alpha + img.cuda() * (1 - alpha)
#         probs = outputs[filter_ids_2].softmax(dim=-1) * alpha + probs.cuda() * (1 - alpha)
#         outputs2, _ = self.model(x[filter_ids_2])
#         loss2 = F.kl_div(outputs2.log_softmax(dim=-1), probs, reduction='batchmean')
#     entropys = entropys[filter_ids_1]
#     loss = entropys.mul(1 / (torch.exp(entropys.clone().detach() - self.e_margin))).mean()
#     if self.memory_bank.num_features_stored > 2000:
#         loss += 0.3 * loss2

#     # Add info_nce loss
#     if self.memory_bank.num_features_stored > 2000:
#         query = features[filter_ids_2]
#         positive_key = features[filter_ids_1]
#         loss += info_nce(query, positive_key)

#     loss.backward()
#     self.optimizer.step()
#     self.optimizer.zero_grad()
#     return outputs, entropys.size(0), filter_ids_1[0].size(0), None