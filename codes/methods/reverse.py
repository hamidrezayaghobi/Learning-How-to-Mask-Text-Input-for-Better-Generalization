from .method import Method

class Reverse(Method):
    def __init__(self, method_name):
        super(Reverse, self).__init__(method_name)

    def execute(self, input_ids, mask, attention_mask, predicted_attention_mask):
        reversed_attention_mask = 1 - mask
        reversed_attention_mask[:, 0] = 1
        reversed_attention_mask[:, -1] = 1

        return input_ids, mask, reversed_attention_mask
