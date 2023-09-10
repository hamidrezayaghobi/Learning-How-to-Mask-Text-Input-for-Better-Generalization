from .method import Method

import torch

class RationalTopTokenReplacing(Method):
    def __init__(self, method_name, k, use_grad_cam,
                 augmentation=False, label_replacing=False):
        super(RationalTopTokenReplacing, self).__init__(method_name)
        self.k = k
        self.use_grad_cam = use_grad_cam
        self.augmentation = augmentation
        self.label_replacing = label_replacing

    def _replace(self, input_ids, replacing_attention_mask, labels):
        replacing_labels = labels.clone()
        batch_size = input_ids.shape[0]
        replacing_attention_mask = 1 - replacing_attention_mask
        replacing_attention_mask[:, 0] = 1
        replacing_attention_mask[:, -1] = 1
        replacing_attention_mask = replacing_attention_mask.int()
        epsilon = 0.01

        input_ids = input_ids + epsilon

        permitted_to_replaceing = torch.sum(replacing_attention_mask == 0, dim=1) > 0

        valid_attention_mask_to_find_min_k = replacing_attention_mask[permitted_to_replaceing]
        if len(valid_attention_mask_to_find_min_k) == 0:
            input_ids = input_ids - epsilon
            input_ids = input_ids.long()
            return input_ids, replacing_labels
        min_k = min((valid_attention_mask_to_find_min_k == 0).count_nonzero(dim=1))

        replacing_attention_mask[(replacing_attention_mask == 0).cumsum(dim=1) > min_k.item()] = 1

        replacing_input_ids = input_ids * (1 - replacing_attention_mask)
        fixed_input_ids = input_ids * replacing_attention_mask

        perm = torch.randperm(len(valid_attention_mask_to_find_min_k))

        if self.label_replacing:
            replacing_labels = labels.clone()
            permitted_to_replacing_labels = replacing_labels[permitted_to_replaceing]
            shuffeld_replacing_labels = permitted_to_replacing_labels[perm]
            replacing_labels[permitted_to_replaceing] = shuffeld_replacing_labels

        non_zero_replacing_input_ids = replacing_input_ids[replacing_input_ids != 0].view(len(valid_attention_mask_to_find_min_k), -1)
        shuffled_replacing_input_ids = non_zero_replacing_input_ids[perm]

        input_ids[fixed_input_ids == 0] = shuffled_replacing_input_ids.flatten()
        input_ids = input_ids - epsilon
        input_ids = input_ids.long()
        return input_ids, replacing_labels


    def _split_data(self, tensor, replacing_attention_mask):
        permitted_mask = torch.sum(replacing_attention_mask == 1, dim=1) > self.k
        fixed_tensor = tensor[~permitted_mask]
        flexible_tensor = tensor[permitted_mask]
        return flexible_tensor, fixed_tensor


    def execute(self, input_ids, mask, attention_mask, labels, debug=False, groups=None):
        mask = self.fix_inputs_with_tokens_less_than_k(input_ids, attention_mask, mask, self.k)
        replacing_attention_mask = attention_mask * mask

        more_than_k_input_ids, less_than_k_input_ids = self._split_data(input_ids, replacing_attention_mask)
        more_than_k_labels, less_than_k_labels = self._split_data(labels, replacing_attention_mask)
        more_than_k_replacing_attention_mask, less_than_k_replacing_attention_mask = self._split_data(replacing_attention_mask, replacing_attention_mask)

        replaced_input_ids, replacing_labels = self._replace(more_than_k_input_ids, more_than_k_replacing_attention_mask, more_than_k_labels)
        less_than_k_replaced_input_ids, less_than_k_replacing_labels = self._replace(less_than_k_input_ids, less_than_k_replacing_attention_mask, less_than_k_labels)

        replaced_input_ids = torch.cat([less_than_k_replaced_input_ids, replaced_input_ids])
        replacing_labels = torch.cat([less_than_k_replacing_labels, replacing_labels])

        if self.augmentation:
            augmentation_indices = replacing_labels != labels

            augmentation_input_ids = input_ids[augmentation_indices]
            augmentation_replaced_input_ids = replaced_input_ids[augmentation_indices]
            replaced_input_ids = torch.cat([input_ids, augmentation_replaced_input_ids])
            input_ids = torch.cat([input_ids, augmentation_input_ids])


            augmentation_labels = labels[augmentation_indices]
            augmentation_replacing_labels = replacing_labels[augmentation_indices]
            replacing_labels = torch.cat([labels, augmentation_replacing_labels])
            labels = torch.cat([labels, augmentation_labels])

            augmentation_mask = mask[augmentation_indices]
            mask = torch.cat([mask, augmentation_mask])

            augmentation_attention_mask = attention_mask[augmentation_indices]
            attention_mask = torch.cat([attention_mask, augmentation_attention_mask])

        if debug:
            if self.augmentation:
                augmentation_groups = groups[augmentation_indices.cpu()]
                groups = torch.cat([groups, augmentation_groups])

            input_ids = torch.cat([less_than_k_input_ids, more_than_k_input_ids])
            labels = torch.cat([less_than_k_labels, more_than_k_labels])

            visualize(input_ids, replaced_input_ids, groups, labels, replacing_labels)

        return replaced_input_ids, mask, replacing_labels, attention_mask