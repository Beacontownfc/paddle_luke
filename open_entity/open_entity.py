import paddle.nn as nn
import paddle.nn.functional as F
import paddle

from luke_model.model import LukeEntityAwareAttentionModel


class LukeForEntityTyping(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__(args.model_config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        weight_attr, bias_attr = None, None
        if args.do_train:
            weight_attr = paddle.ParamAttr(name="weight", initializer=paddle.nn.initializer.Normal(mean=0.0, 
                                                std=args.model_config.initializer_range))
            bias_attr = paddle.ParamAttr(name="bias", initializer=paddle.nn.initializer.Constant(value=0.0))
        self.typing = nn.Linear(args.model_config.hidden_size, num_labels, weight_attr=weight_attr, bias_attr=bias_attr)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        labels=None,
    ):
        encoder_outputs = super(LukeForEntityTyping, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        feature_vector = encoder_outputs[1][:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits

        return (F.binary_cross_entropy_with_logits(logits.reshape([-1]), labels.reshape([-1]).astype('float32')),)