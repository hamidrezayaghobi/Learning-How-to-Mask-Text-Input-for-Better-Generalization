import copy
import torch

def compute_grad_cam(model, input_ids, attention_mask, token_type_ids=None,
                     position_ids=None, head_mask=None, inputs_embeds=None):

    copied_model = copy.deepcopy(model).to(DEVICE)

    outputs = copied_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
    logits = outputs['logits']

    # Backpropagate to get the gradients
    target_class = torch.argmax(logits, dim=1)
    one_hot = torch.zeros_like(logits).scatter(1, target_class.unsqueeze(1), 1.0)
    logits.backward(gradient=one_hot, retain_graph=True)

    # Get the gradients for each token in the input text
    # token_ids = inputs['input_ids']
    gradients = copied_model.bert_model.embeddings.word_embeddings.weight.grad[input_ids] #shape = [number of text, number of tokens, 768]
    gradients = torch.mean(gradients, dim=2)  # Aggregate gradients across layers   shape = [number of text, number of tokens]
    gradients = abs(gradients)
    return gradients