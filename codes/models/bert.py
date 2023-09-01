import torch.nn as nn
from transformers import BertTokenizer, BertModel

from .config import PRE_TRAINED_MODEL_NAME

class Bert(nn.Module):
    def __init__(self, num_labels, tune_only_last_layer=True):
        super(Bert, self).__init__()

        #Pre Trained Bert
        self.bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        #Freezing Layers
        if tune_only_last_layer:
            for name, param in self.bert_model.named_parameters():
                if 'classifier' in name:
                  param.requires_grad = True
                else:
                  param.requires_grad = False

        self.num_labels = num_labels

        #Classification Layer
        self.dropout = nn.Dropout(0.2)
        self.last_layer_classifier = nn.Linear(self.bert_model.config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, use_grad_cam=False,
                rational_replacing=False, rational_augmentation=False, train_agument=False,
                test_mode=False, test_reverse=False, debug=False):

        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        output = self.dropout(outputs[1])
        logits = self.last_layer_classifier(output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}