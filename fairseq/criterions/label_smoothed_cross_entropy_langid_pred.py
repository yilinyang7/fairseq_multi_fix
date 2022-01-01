import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class LabelSmoothedCrossEntropyLangIDPredCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    langid_pred_coef: float = field(
        default=0.1,
        metadata={"help": "weight coefficient for langid prediction loss"}
    )

@register_criterion(
    "label_smoothed_cross_entropy_langid_pred",
    dataclass=LabelSmoothedCrossEntropyLangIDPredCriterionConfig
)
class LabelSmoothedCrossLangIDPredEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.args = task.args
        self.lang_dict = task.data_manager.lang_dict

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        nmt_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        langid_loss = self.compute_langid_pred_loss(model, net_output, sample)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        loss = (1 - self.args.langid_pred_coef) * nmt_loss + \
                self.args.langid_pred_coef * langid_loss
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_langid_pred_loss(self, model, net_output, sample):
        dec_features = net_output[1]['inner_states'][-1].transpose(0, 1).clone()
        target = sample["target"].view(-1)
        langid_pred = model.langid_layer(dec_features, sample['target'])
        langid_labels = sample['tgt_lang_id'].expand_as(sample['target']) - 1
        nll_loss = F.cross_entropy(
            langid_pred.reshape(target.size(0), -1),
            langid_labels.reshape(-1),
            reduction='none',
        )
        pad_mask = target.eq(self.padding_idx)
        nll_loss.masked_fill_(pad_mask, 0.0)
        return nll_loss.sum()

