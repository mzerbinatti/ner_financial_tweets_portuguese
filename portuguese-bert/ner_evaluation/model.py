"""Implementations of BERT, BERT-CRF, BERT-LSTM and BERT-LSTM-CRF models."""

import logging
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Type

import torch
from pytorch_transformers.modeling_bert import (BertConfig,
                                                BertForTokenClassification,
                                                BertEmbeddings,
                                                BertLayerNorm,
                                                BertEncoder,
                                                BertPooler,
                                                BertModel)
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, MSELoss

LOGGER = logging.getLogger(__name__)

# BertForTokenClassificationWithPOS extracted from: https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_bert.html
# pytorch_transformers 1.1.0
class BertForTokenClassificationWithPOS(BertForTokenClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModelWithPOS(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, pos_label_ids=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, pos_label_ids=pos_label_ids)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


def sum_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Sums the last 4 hidden representations of a sequence output of BERT.
    Args:
    -----
    sequence_output: Tuple of tensors of shape (batch, seq_length, hidden_size).
        For BERT base, the Tuple has length 13.

    Returns:
    --------
    summed_layers: Tensor of shape (batch, seq_length, hidden_size)
    """
    last_layers = sequence_outputs[-4:]
    return torch.stack(last_layers, dim=0).sum(dim=0)


def get_last_layer(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Returns the last tensor of a list of tensors."""
    return sequence_outputs[-1]


def concat_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Concatenate the last 4 tensors of a tuple of tensors."""
    last_layers = sequence_outputs[-4:]
    return torch.cat(last_layers, dim=-1)


POOLERS = {
    'sum': sum_last_4_layers,
    'last': get_last_layer,
    'concat': concat_last_4_layers,
}


def get_model_and_kwargs_for_args(
        args: Namespace,
        training: bool = True,
) -> Tuple[Type[torch.nn.Module], Dict[str, Any]]:
    """Given the parsed arguments, returns the correct model class and model
    args.

    Args:
        args: a Namespace object (from parsed argv command).
        training: if True, sets a high initialization value for classifier bias
            parameter after model initialization.
    """
    bias_O = 6 if training else None
    model_args = {
        'pooler': args.pooler,
        'bias_O': bias_O,
    }

    if args.freeze_bert:
        # Possible models: BERT-LSTM or BERT-LSTM-CRF
        model_args['lstm_layers'] = args.lstm_layers
        model_args['lstm_hidden_size'] = args.lstm_hidden_size
        if args.no_crf:
            model_class = BertLSTM
        else:
            model_class = BertLSTMCRF

    else:
        # Possible models: BertForNERClassification or BertCRF
        if args.no_crf:
            model_class = BertForNERClassification
        else:
            model_class = BertCRF

    return model_class, model_args


class BertForNERClassification(BertForTokenClassificationWithPOS):
    """BERT model for NER task.

    The number of NER tags should be defined in the `BertConfig.num_labels`
    attribute.

    Args:
        config: BertConfig instance to build BERT model.
        weight_O: loss weight value for "O" tags in CrossEntropyLoss.
        bias_O: optional value to initiate the classifier's bias value for "O"
            tag.
        pooler: which pooler configuration to use to pass BERT features to the
            classifier.
    """

    def __init__(self,
                 config: BertConfig,
                 weight_O: float = 0.01,
                 bias_O: Optional[float] = None,
                 pooler='last'):
        super().__init__(config)
        del self.classifier  # Deletes classifier of BertForTokenClassification

        # use_pos_embeddings =  kwargs.get('with_pos', False)
        # if use_pos_embeddings:
        self.embeddings = BertEmbeddingsWithPOS(config) # TODO: Fazer parametrizavel. Fixo para testes.

        num_labels = config.num_labels

        if pooler not in POOLERS:
            message = ("Invalid pooler: %s. Pooler must be one of %s."
                       % (pooler, list(POOLERS.keys())))
            raise ValueError(message)

        self._build_classifier(config, pooler)
        if bias_O is not None:
            self.set_bias_tag_O(bias_O)

        assert isinstance(weight_O, float) and 0 < weight_O < 1
        weights = [1.] * num_labels
        weights[0] = weight_O
        weights = torch.tensor(weights)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

        self.frozen_bert = False
        self.pooler = POOLERS.get(pooler)

    def _build_classifier(self, config, pooler):
        """Build tag classifier."""
        if pooler in ('last', 'sum'):
            self.classifier = torch.nn.Linear(config.hidden_size,
                                              config.num_labels)
        else:
            assert pooler == 'concat'
            self.classifier = torch.nn.Linear(4 * config.hidden_size,
                                              config.num_labels)

    def set_bias_tag_O(self, bias_O: Optional[float] = None):
        """Increase tag "O" bias to produce high probabilities early on and
        reduce instability in early training."""
        if bias_O is not None:
            LOGGER.info('Setting bias of OUT token to %s.', bias_O)
            self.classifier.bias.data[0] = bias_O

    def freeze_bert(self):
        """Freeze all BERT parameters. Only the classifier weights will be
        updated."""
        for p in self.bert.parameters():
            p.requires_grad = False
        self.frozen_bert = True

    def bert_encode(self, input_ids, token_type_ids=None, attention_mask=None, pos_label_ids=None):
        """Gets encoded sequence from BERT model and pools the layers accordingly.
        BertModel outputs a tuple whose elements are:
        1- Last encoder layer output. Tensor of shape (B, S, H)
        2- Pooled output of the [CLS] token. Tensor of shape (B, H)
        3- Encoder inputs (embeddings) + all Encoder layers' outputs. This
            requires the flag `output_hidden_states=True` on BertConfig. Returns
            List of tensors of shapes (B, S, H).
        4- Attention results, if `output_attentions=True` in BertConfig.

        This method uses just the 3rd output and pools the layers.
        """
        _, _, all_layers_sequence_outputs, *_ = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pos_label_ids=pos_label_ids)

        # Use the defined pooler to pool the hidden representation layers
        sequence_output = self.pooler(all_layers_sequence_outputs)

        return sequence_output

    def predict_logits(self, input_ids, token_type_ids=None,
                       attention_mask=None, pos_label_ids=None):
        """Returns the logits prediction from BERT + classifier."""
        if self.frozen_bert:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(
                input_ids, token_type_ids, attention_mask, pos_label_ids=pos_label_ids)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq, tags)

        return logits

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                pos_label_ids=None
                ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the loss.
        Otherwise, it will return the logits and predicted tags tensors.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns a dict with calculated tensors:
          - "logits"
          - "y_pred"
          - "loss" (if `labels` is not `None`)
        """
        outputs = {}

        logits = self.predict_logits(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     pos_label_ids=pos_label_ids)
        _, y_pred = torch.max(logits, dim=-1)
        y_pred = y_pred.cpu().numpy()
        outputs['logits'] = logits
        outputs['y_pred'] = y_pred

        if labels is not None:
            # Only keep active parts of the loss
            mask = prediction_mask
            if mask is not None:
                mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss

        return outputs


class BertCRF(BertForNERClassification):
    """BERT-CRF model.

    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        del self.loss_fct  # Delete unused CrossEntropyLoss
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                pos_label_ids=None
                ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` is not `None`, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns a dict with calculated tensors:
          - "logits"
          - "loss" (if `labels` is not `None`)
          - "y_pred" (if `labels` is `None`)
        """
        outputs = {}

        logits = self.predict_logits(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     pos_label_ids=pos_label_ids)
        outputs['logits'] = logits

        # mask: mask padded sequence and also subtokens, because they must
        # not be used in CRF.
        mask = prediction_mask
        batch_size = logits.shape[0]

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss

        else:
            # Same reasons for iterating
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                output_tags.append(tags[0])

            outputs['y_pred'] = output_tags

        return outputs


class BertLSTM(BertForNERClassification):
    """BERT model with an LSTM model as classifier. This model is meant to be
    used with frozen BERT schemes (feature-based).

    Args:
        config: BertConfig instance to build BERT model.
        lstm_hidden_size: hidden size of LSTM layers. Defaults to 100.
        lstm_layers: number of LSTM layers. Defaults to 1.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self,
                 config: BertConfig,
                 lstm_hidden_size: int = 100,
                 lstm_layers: int = 1,
                 **kwargs: Any):

        lstm_dropout = 0.2 if lstm_layers > 1 else 0
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        pooler = kwargs.get('pooler', 'last')

        super().__init__(config, **kwargs)

        if pooler in ('last', 'sum'):
            lstm_input_size = config.hidden_size
        else:
            assert pooler == 'concat'
            lstm_input_size = 4 * config.hidden_size

        self.lstm = torch.nn.LSTM(input_size=lstm_input_size,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=lstm_layers,
                                  dropout=lstm_dropout,
                                  batch_first=True,
                                  bidirectional=True)

    def _build_classifier(self, config, pooler):
        """Build label classifier."""
        self.classifier = torch.nn.Linear(2 * self.lstm_hidden_size,
                                          config.num_labels)

    def _pack_bert_encoded_sequence(self, encoded_sequence, attention_mask):
        """Returns a PackedSequence to be used by LSTM.

        The encoded_sequence is the output of BERT, of shape (B, S, H).
        This method sorts the tensor by sequence length using the
        attention_mask along the batch dimension. Then it packs the sorted
        tensor.

        Args:
        -----
        encoded_sequence (tensor): output of BERT. Shape: (B, S, H)
        attention_mask (tensor): Shape: (B, S)

        Returns:
        --------
        sorted_encoded_sequence (tensor): sorted `encoded_sequence`.
        sorted_ixs (tensor): tensor of indices returned by `torch.sort` when
            performing the sort operation. These indices can be used to unsort
            the output of the LSTM.
        """
        seq_lengths = attention_mask.sum(dim=1)   # Shape: (B,)
        sorted_lengths, sort_ixs = torch.sort(seq_lengths, descending=True)

        sorted_encoded_sequence = encoded_sequence[sort_ixs, :, :]

        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_encoded_sequence,
            sorted_lengths,
            batch_first=True)

        return packed_sequence, sort_ixs

    def _unpack_lstm_output(self, packed_sequence, sort_ixs):
        """Unpacks and unsorts a sorted PackedSequence that is output by LSTM.

        Args:
            packed_sequence (PackedSequence): output of LSTM. Shape: (B, S, Hl)
            sort_ixs (tensor): the indexes of be used for unsorting. Shape: (B,)

        Returns:
            The unsorted sequence.
        """
        B = len(sort_ixs)

        # Unpack
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence,
                                                             batch_first=True)

        assert unpacked.shape <= (B, 512, 2 * self.lstm.hidden_size)

        # Prepare indices for unsort
        sort_ixs = sort_ixs.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
        # (B, S, Hl)
        sort_ixs = sort_ixs.expand(-1, unpacked.shape[1], unpacked.shape[2])
        # Unsort
        unsorted_sequence = (torch.zeros_like(unpacked)
                             .scatter_(0, sort_ixs, unpacked))

        return unsorted_sequence

    def forward_lstm(self, bert_encoded_sequence, attention_mask):
        packed_sequence, sorted_ixs = self._pack_bert_encoded_sequence(
            bert_encoded_sequence, attention_mask)

        packed_lstm_out, _ = self.lstm(packed_sequence)
        lstm_out = self._unpack_lstm_output(packed_lstm_out, sorted_ixs)

        return lstm_out

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, prediction_mask=None):
        """Performs the forward pass of the network.

        Computes the logits, predicted tags and if `labels` is not None, it will
        it will calculate and return the the loss, that is, the negative
        log-likelihood of the batch.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns:
            A dict with calculated tensors:
            - "logits"
            - "y_pred"
            - "loss" (if `labels` is not `None`)
        """
        outputs = {}

        if self.frozen_bert:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(
                input_ids, token_type_ids, attention_mask)

        sequence_output = self.dropout(sequence_output)  # (batch, seq, H)

        lstm_out = self.forward_lstm(
            sequence_output, attention_mask)  # (batch, seq, Hl)
        sequence_output = self.dropout(lstm_out)

        logits = self.classifier(sequence_output)
        _, y_pred = torch.max(logits, dim=-1)
        y_pred = y_pred.cpu().numpy()
        outputs['logits'] = logits
        outputs['y_pred'] = y_pred

        if labels is not None:
            # Only keep active parts of the loss
            mask = prediction_mask
            if mask is not None:
                # Adjust mask and labels to have the same length as logits
                mask = mask[:, :logits.size(1)].contiguous()
                labels = labels[:, :logits.size(1)].contiguous()

                mask = mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[mask]
                active_labels = labels.view(-1)[mask]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

            outputs['loss'] = loss

        return outputs


class BertLSTMCRF(BertLSTM):
    """BERT model with an LSTM-CRF as classifier. This model is meant to be
    used with frozen BERT schemes (feature-based).

    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass (see BertLSTM).
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                prediction_mask=None,
                ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the network.

        If `labels` are not None, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Otherwise, it will calculate the most probable sequence outputs using
        Viterbi decoding and return a list of sequences (List[List[int]]) of
        variable lengths.

        Args:
            input_ids: tensor of input token ids.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).

        Returns:
            A dict with calculated tensors:

            - "logits"
            - "loss" (if `labels` is not `None`)
            - "y_pred" (if `labels` is `None`)
        """
        outputs = {}

        if self.frozen_bert:
            sequence_output = input_ids
        else:
            sequence_output = self.bert_encode(
                input_ids, token_type_ids, attention_mask)

        sequence_output = self.dropout(sequence_output)  # (batch, seq, H)

        lstm_out = self.forward_lstm(
            sequence_output, attention_mask)  # (batch, seq, Hl)
        sequence_output = self.dropout(lstm_out)
        logits = self.classifier(sequence_output)
        outputs['logits'] = logits

        mask = prediction_mask  # (B, S)
        # Logits sequence length depends on the inputs:  logits.shape <= (B, S)
        # We have to make the mask and labels the same size.
        mask = mask[:, :logits.size(1)].contiguous()

        if labels is not None:
            # Negative of the log likelihood.
            # Loop through the batch here because of 2 reasons:
            # 1- the CRF package assumes the mask tensor cannot have interleaved
            # zeros and ones. In other words, the mask should start with True
            # values, transition to False at some moment and never transition
            # back to True. That can only happen for simple padded sequences.
            # 2- The first column of mask tensor should be all True, and we
            # cannot guarantee that because we have to mask all non-first
            # subtokens of the WordPiece tokenization.
            labels = labels[:, :logits.size(1)].contiguous()
            batch_size = input_ids.size(0)
            loss = 0
            for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                seq_labels = seq_labels[seq_mask].unsqueeze(0)
                loss -= self.crf(seq_logits, seq_labels,
                                 reduction='token_mean')

            loss /= batch_size
            outputs['loss'] = loss

        else:
            # Same reasons for iterating
            output_tags = []
            for seq_logits, seq_mask in zip(logits, mask):
                seq_logits = seq_logits[seq_mask].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                output_tags.append(tags[0])

            outputs['y_pred'] = output_tags

        return outputs


# https://discuss.huggingface.co/t/how-to-use-additional-input-features-for-ner/4364/27
class BertEmbeddingsWithPOS(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        # print("###################### config.num_pos_labels ######################")
        # print(config.num_pos_labels)
        # 17 classes + CLS (0 ... 17)
        max_number_of_pos_label = 17 + 1 # TODO: Deixar dinamico. Fixo para testes (WIP)
        self.pos_embeddings = torch.nn.Embedding(max_number_of_pos_label, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, pos_label_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if pos_label_ids is not None:
            pos_label_embeddings = self.pos_embeddings(pos_label_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings + pos_label_embeddings
        else:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# BertModel extracted from: https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_bert.html
# pytorch_transformers 1.1.0
class BertModelWithPOS(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddingsWithPOS(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_weights)

    # def _resize_token_embeddings(self, new_num_tokens):
    #     old_embeddings = self.embeddings.word_embeddings
    #     new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    #     self.embeddings.word_embeddings = new_embeddings
    #     return self.embeddings.word_embeddings

    # def _prune_heads(self, heads_to_prune):
    #     """ Prunes heads of the model.
    #         heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
    #         See base class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, pos_label_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if pos_label_ids is None:
            pos_label_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, pos_label_ids=pos_label_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions) 


