import numpy as np
import os
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.notebook import tqdm


def run_epoch(run_type, device, data_set_type, model, data_loader, optimizer, scheduler=None, **kwargs):
    assert run_type in ['Train', 'Test', 'Val'], print("Error! undefined run_epoch type")
    if 'debug' not in kwargs:
        kwargs['debug'] = False

    pbar = tqdm(data_loader)

    accuracy = 0
    epoch_loss = 0
    if run_type in ['Test', 'Val']:
        model.eval()
    else:
        model.train()
        optimizer.zero_grad()

    all_preds = []
    all_labels = []
    all_groups = []

    with torch.set_grad_enabled(run_type == 'Train'):
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            groups = batch['groups']
            segments_ids = batch['segments_ids'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                            token_type_ids=segments_ids,
                            use_grad_cam=kwargs['use_grad_cam'],
                            rational_replacing=kwargs['rational_replacing'],
                            rational_augmentation=kwargs['rational_augmentation'],
                            train_label_replacing=kwargs['label_replacing'],
                            train_agument=kwargs['agument'],
                            test_reverse = kwargs['test_reverse'],
                            test_mode = kwargs['test_not_masking'],
                            debug=kwargs['debug'])

            loss = outputs['loss']
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            if 'labels' in outputs:
                labels = outputs['labels']

            if run_type == 'Train':
                if data_set_type == 'MultiNLI':
                    max_grad_norm = 1.0
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    # model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss += loss.item()

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            accuracy += accuracy_score(labels, preds)

            avg_loss = epoch_loss / ((batch_idx + 1))
            avg_accuracy = accuracy / ((batch_idx + 1))
            pbar.set_description(f"{run_type} => AvgLoss:{avg_loss:.4f}, AvgAcc:{avg_accuracy:.4f}")

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_groups.extend(groups.numpy())

    if run_type == 'Test':
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_groups = np.array(all_groups)

        for group in np.unique(all_groups):
            mask = all_groups == group
            group_labels = all_labels[mask]
            group_preds = all_preds[mask]
            precision, recall, f1, _ = precision_recall_fscore_support(group_labels, group_preds, average='micro')
            print(f"group={group} ==> acc={precision} - count={sum(mask)}")

    data_loader_len = len(data_loader)
    epoch_loss /= data_loader_len
    accuracy /= data_loader_len

    return model, loss, accuracy


def save_model(model, model_name, epoch, lr, batch_size, data_set_type, max_length, k=None):
    if k:
        model_path = f"./models/dataset={data_set_type}/max_length={max_length}/{model_name}/epoch={epoch}_lr={lr}_batch_size={batch_size}_k={k}.pt"
    else:
        model_path = f"./models/dataset={data_set_type}/max_length={max_length}/{model_name}/epoch={epoch}_lr={lr}_batch_size={batch_size}.pt"

    if os.path.exists(model_path):
        print("WARNING: Model already exist!, nothing saved")
        return None

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    torch.save(model.state_dict(), model_path)
    print("model saved in this path:")
    print(model_path)