from paddle.io import Dataset
import numpy as np

class DataGenerator(Dataset):
    def __init__(self, features, args):
        super(DataGenerator, self).__init__()
        self.args = args
        self.all_word_ids = [f.word_ids for f in features]
        self.all_word_segment_ids = [f.word_segment_ids for f in features]
        self.all_word_attention_mask = [f.word_attention_mask for f in features]
        self.all_entity_ids = [f.entity_ids for f in features]
        self.all_entity_position_ids = [f.entity_position_ids for f in features]
        self.all_entity_segment_ids = [f.entity_segment_ids for f in features]
        self.all_entity_attention_mask = [f.entity_attention_mask for f in features]
        self.all_labels = [f.labels for f in features]

    def __getitem__(self, item):
        word_ids = self.all_word_ids[item]
        word_segment_ids = self.all_word_segment_ids[item]
        word_attention_mask = self.all_word_attention_mask[item]
        entity_ids = self.all_entity_ids[item]
        entity_position_ids = self.all_entity_position_ids[item]
        entity_segment_ids = self.all_entity_segment_ids[item]
        entity_attention_mask = self.all_entity_attention_mask[item]
        label = self.all_labels[item]

        return word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, \
               entity_segment_ids, entity_attention_mask, label
    
    def __len__(self):
        return len(self.all_word_ids)

    def pad_sequence(self, value, padding_value, max_len, flag=False):
        if flag:
            if len(value) > max_len:
                return np.array(value[:max_len], np.int64)
            else:
                res = value + [[padding_value] * len(value[0])] * (max_len - len(value))
                return np.array(res, np.int64)
        else:
            if len(value) > max_len:
                return np.array(value[:max_len], np.int64)
            else:
                return np.array(value + [padding_value] * (max_len - len(value)), np.int64)
