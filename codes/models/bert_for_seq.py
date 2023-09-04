import torch.nn as nn
from pytorch_transformers import BertConfig, BertForSequenceClassification

from .config import PRE_TRAINED_MODEL_NAME

class BertForSequence(nn.Module):
    def __init__(self, num_labels, tune_only_last_layer=True, finetuning_task="mnli"):
        super(BertForSequence, self).__init__()

        #Pre Trained Bert
        config = BertConfig.from_pretrained(
          'bert-base-uncased',
          num_labels=num_labels,
          finetuning_task=finetuning_task)

        self.bert_model = BertForSequenceClassification.from_pretrained(
          'bert-base-uncased',
          from_tf=False,
          config=config
          )

        

        #Freezing Layers
        if tune_only_last_layer:
            for name, param in self.bert_model.named_parameters():
                if 'classifier' in name:
                  param.requires_grad = True
                else:
                  param.requires_grad = False

        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, use_grad_cam=False,
                rational_replacing=False, rational_augmentation=False,
                train_agument=False, train_label_replacing=False,
                test_mode=False, test_reverse=False, debug=False, groups=None):

        outputs = self.bert_model(input_ids=input_ids, 
                                  attention_mask=attention_mask, 
                                  token_type_ids=token_type_ids, 
                                  head_mask=head_mask, 
                                  labels=labels)

        logits = outputs[1]

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}