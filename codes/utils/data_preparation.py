import ast
import os
import pickle
import torch

def creat_spurious_pdf(pdf, prob, data_type, data_set, data_set_type):
    def add_spurious(row):
        if row['prob'] < prob:
            if row[data_set[data_set_type]['label']] == 1:
                place = random.randint(0, MAX_LENGTH / 2)
                return " ".join(row[data_set[data_set_type]['text']].split()[:place]) + ' Hamid ' + " ".join(row[data_set[data_set_type]['text']].split()[place:])
                # return 'Hamid Hamid Hamid Hamid Hamid ' + row[data_set[DATA_SET_TYPE]['text']]
            else:
                place = random.randint(0, MAX_LENGTH / 2)
                return " ".join(row[data_set[data_set_type]['text']].split()[:place]) + ' Akbar ' + " ".join(row[data_set[data_set_type]['text']].split()[place:])
                # return 'Akbar Akbar Akbar Akbar Akbar ' + row[data_set[DATA_SET_TYPE]['text']]
        else:
            if row[data_set[data_set_type]['label']] == 1:
                place = random.randint(0, MAX_LENGTH / 2)
                return " ".join(row[data_set[data_set_type]['text']].split()[:place]) + ' Akbar ' + " ".join(row[data_set[data_set_type]['text']].split()[place:])
                # return 'Akbar Akbar Akbar Akbar Akbar ' + row[data_set[DATA_SET_TYPE]['text']]
            else:
                place = random.randint(0, MAX_LENGTH / 2)
                return " ".join(row[data_set[data_set_type]['text']].split()[:place]) + ' Hamid ' + " ".join(row[data_set[data_set_type]['text']].split()[place:])
                # return 'Hamid Hamid Hamid Hamid Hamid ' + row[data_set[DATA_SET_TYPE]['text']]

    if data_type == 'train':
        prob = 0.7
    else:
        prob = 0.3
        
    pdf['prob'] = np.random.random(len(pdf))
    pdf[data_set[data_set_type]['text']] = pdf.apply(add_spurious, axis=1)
    return pdf
    

def prepare_segments(pdf, max_length):
    pdf['segments'] = pdf['segments'].apply(lambda x: x.replace('\n', ''))
    pdf['segments'] = pdf['segments'].apply(lambda x: x.replace(' ', ', '))
    pdf['segments'] = pdf['segments'].apply(ast.literal_eval)
    pdf['segments'] = pdf['segments'].apply(lambda x: x[:max_length])
    return pdf
    

def tokenize_dataset(tokenizer, text, split, data_set_type, max_length):
    tokenized_path = f"tokenized_dataset/{data_set_type}_max_length={max_length}_{split}.json"
    print(f"Data Type = {data_set_type}")
    
    if os.path.exists(tokenized_path):
        print(f'Loading Tokenized {split} Data ...')
        with open(tokenized_path, "rb") as json_file:
            encodings = pickle.load(json_file)
        print(f"Tokenized {split} Data Loaded")
    else:
        print(f"Tokenizing {split} Data ...")
        encodings = tokenizer(text, truncation=True, padding=True, max_length=max_length)
        print(f"Saving Tokenized {split} Data ...")
        with open(tokenized_path, "wb") as json_file:
            pickle.dump(encodings, json_file)
        print(f"Tokenized {split} Data Saved")

    return encodings

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, groups, segment_ids):
        self.encodings = encodings
        self.labels = labels
        self.groups = groups
        self.segment_ids = segment_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        item['groups'] = torch.tensor(int(self.groups[idx]))
        item['segments_ids'] = torch.tensor(self.segment_ids[idx])
        return item

    def __len__(self):
        return len(self.labels)