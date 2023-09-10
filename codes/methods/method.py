import torch

class Method():
    def __init__(self, method_name):
        self.method_name = method_name

    @staticmethod
    def fix_inputs_with_tokens_less_than_k(input_ids, attention_mask, mask, k):
        acceptable_inputs_indices = torch.sum(attention_mask == 1, dim=1)  > (k + 2) # +2 => one is CLS and one is SEP
        mask[~acceptable_inputs_indices] = 0
        return mask

    @staticmethod
    def cal_continuity_loss(z):
        return torch.mean(torch.abs(z[:, 1:] - z[:, :-1]))

    def execute(self, input_ids, attention_mask, predicted_attention_mask, debug=False):
        return NotImplementedError("Methods execute function must be implemented")