import datetime
import logging
import time
import contextlib
import numpy as np
from collections import defaultdict
from itertools import chain
import torch
from fairseq import utils
import torch.nn.functional as F
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    ConcatDataset,
    data_utils,
    iterators,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.utils import FileContentsAction
import fairseq.distributed.utils as distributed_utils

logger = logging.getLogger(__name__)

def dot_product(g_i, g_j): # use double() if returns INF
    return torch.dot(g_i.float(), g_j.float()).item()

def l2_norm(g_i):
    return g_i.float().norm().item()

def pcgrad_proj(g_i, g_j):
    return (dot_product(g_i, g_j) / (l2_norm(g_j) ** 2)) * g_j

@register_task("translation_multi_tgp")
class TranslationMultiTGPTask(TranslationMultiSimpleEpochTask):
    """
    Multilingual Translation with TGP (Target-Language-Projection)
    based on TranslationMultiSimpleEpochTask

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not
    """
    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.langs = args.langs
        self.oracle_update = 0
        self.tgp_scope = args.tgp_scope
        self.retrieve_every = args.tgp_retrieve_every
        self.max_tokens = args.tgp_max_tokens
        self.group_langs = set(args.tgp_group_langs.split(','))
        self.group_langs.discard('')
        self._oracle_grad = None

        orcl_lang_pairs = ["oracle_%s-%s" % (l, l) for l in self.langs]
        self.oracle_manager = MultilingualDatasetManager.setup_data_manager(
            args, orcl_lang_pairs, langs, defaultdict(lambda: dicts['en']), None
        )

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--tgp-scope', type=str, default='model', choices=['model', 'layer', 'module'])
        parser.add_argument('--tgp-retrieve-every', type=int, default=200)
        parser.add_argument('--tgp-group-langs', type=str, default='')
        parser.add_argument('--tgp-max-tokens', type=int, default=4096)
        TranslationMultiSimpleEpochTask.add_args(parser)
        # fmt: on

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == 'train':
            return super().load_dataset(split, epoch, combine, **kwargs)

        # load valid/oracle data
        self.datasets['valid'] = self.oracle_manager.load_dataset(
            "test", False, 0, combine=False, shard_epoch=None,
        )
        self.datasets['oracle'] = {
            k.split('-')[-1]: d for k, d in self.oracle_manager.load_split_datasets(
                "valid", False, 0, combine=False, shard_epoch=None,
            )[0]
        }
        if len(self.group_langs):
            self.datasets['oracle']['others'] = ConcatDataset([
                d for k, d in self.datasets['oracle'].items() if k in self.group_langs
            ])
            for lang in self.group_langs:
                del self.datasets['oracle'][lang]

        self.oracle_langs = list(self.datasets['oracle'].keys())
        self.lang2id = {l: i for i, l in enumerate(self.oracle_langs)}


    def _prepare_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        sample = utils.move_to_cuda(sample)
        if self.args.fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        return sample


    @property
    def dummy_batch(self): # on CPU
        ret = {'net_input': {}}
        ret['id'] = torch.arange(10)
        ret['nsentences'] = 10
        ret['ntokens'] = 100
        ret['target'] = torch.LongTensor(10, 10).fill_(1)
        ret['net_input']['src_tokens'], ret['net_input']['prev_output_tokens'] = ret['target'], ret['target']
        ret['net_input']['src_lengths'] = torch.LongTensor(10).fill_(10)
        ret['is_dummy_batch'] = True
        return ret


    def oracle_grad(self, update_num, model, criterion, optimizer):
        if self._oracle_grad is not None \
            and (update_num == self.oracle_update
                or (update_num - self.oracle_update) % self.retrieve_every):
            return self._oracle_grad

        # update self._oracle_grad
        model.train()
        criterion.train()
        optimizer.zero_grad()
        fp32_params = optimizer.fp32_params[torch.cuda.current_device()]
        self.oracle_update = update_num
        self._oracle_grad = torch.zeros_like(
            fp32_params.grad, dtype=torch.float16, device=torch.device('cpu')
        ).repeat(len(self.oracle_langs), 1)
        tmp_grad = torch.zeros_like(fp32_params.grad)

        for i, (lang, dataset) in enumerate(self.datasets['oracle'].items()):
            steps = 0
            total_loss, total_nsents, total_sample_size = 0, 0, 0
            itr = self.get_batch_iterator(
                dataset=dataset,
                max_tokens=4096,
                max_positions=utils.resolve_max_positions(
                    self.max_positions(),
                    model.max_positions(),
                ),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_shards=self.args.distributed_world_size,
                shard_id=self.args.distributed_rank,
                num_workers=self.args.num_workers,
            ).next_epoch_itr(shuffle=True)

            for sample in itr:
                if steps >= 100: # reducing time
                    break

                if sample is None or len(sample) == 0:
                    sample = self.dummy_batch
                    ignore_grad = True
                else:
                    ignore_grad = False

                sample = self._prepare_sample(sample)
                loss, sample_size, logging_output = super().train_step(
                    sample, model, criterion, optimizer, ignore_grad
                )
                del loss

                if self.args.distributed_world_size > 1:
                    logging_outputs, sample_sizes = zip(*distributed_utils.all_gather_list(
                        [[logging_output], [sample_size]])
                    )
                    logging_outputs = list(chain.from_iterable(logging_outputs))
                    logging_output = self.aggregate_logging_outputs(logging_outputs, criterion)
                    sample_sizes = list(chain.from_iterable(sample_sizes))
                    sample_size = sum(sample_sizes)

                if sample_size > 0:
                    optimizer.multiply_grads(self.args.distributed_world_size)
                else:
                    logger.info(" | no samples")
                    optimizer.zero_grad()
                    continue

                try:
                    grad_norm = optimizer.clip_grad_norm(self.args.clip_norm)
                except OverflowError as e:
                    logger.info(f"| WARNING: overflow detected, {str(e)}")
                    optimizer.zero_grad()
                    continue

                flatten_grads = fp32_params.grad.detach().clone()
                optimizer.zero_grad()
                updated_grads = steps / (steps + 1) * tmp_grad + flatten_grads / (steps + 1)
                if torch.isnan(updated_grads).any() or torch.isinf(updated_grads).any():
                    logger.info(
                        f"| Warning: overflow detected during gradient retrieve at {lang}"
                        f" {steps} step (process {self.args.distributed_rank}): "
                        f"{torch.isnan(updated_grads).any().item()} "
                        f"{torch.isinf(updated_grads).any().item()}"
                     )
                    continue
                else:
                    total_loss += logging_output['nll_loss']
                    total_nsents += logging_output.get('bsz', logging_output.get('nsentences', 0))
                    total_sample_size += sample_size
                    tmp_grad = updated_grads
                    steps += 1

            self._oracle_grad[i].copy_(tmp_grad.cpu())
            logger.info(
                f"{lang}, {self.args.distributed_rank}, {steps}, nsents: {total_nsents},"
                f"sample_sizes: {total_sample_size}, nll_loss: "
                f"{total_loss / total_sample_size}, mean: {tmp_grad.mean().item()}, "
                f"std: {tmp_grad.std().item()}"
            )

        return self._oracle_grad


    def _flatten_grad(self, model, optimizer, zero_grad=True):
        grad = []
        for p in model.parameters():
            if p.grad is None:
                grad.append(torch.zeros_like(p.view(-1)))
            else:
                grad.append(p.grad.detach().flatten())

        if zero_grad:
            optimizer.zero_grad()
        return torch.cat(grad)


    def set_grad(self, model, optimizer, grads):
        idx = 0
        for p in model.parameters():
            shape = p.shape
            length = np.prod(shape)
            p.grad = grads[idx:idx + length].view(shape).to(p, copy=True)
            idx += length
        optimizer._needs_sync = True


    def split_sample(self, sample):
        """
        Splitting language samples into smaller ones according to
        args.tgp_max_tokens
        """
        ret = []
        for lang, batch in sample.items():
            if lang in self.group_langs:
                lang = "others"
            # real number of tokens
            ntokens = max(
                batch['net_input']['src_tokens'].nelement(),
                batch['target'].nelement(),
            )
            if ntokens <= self.max_tokens:
                ret.append((lang, batch))
            else:
                num_splits = 1 + ntokens // self.max_tokens
                nsents = 1 + batch['nsentences'] // num_splits
                for i in range(num_splits):
                    start, end = i * nsents, (i + 1) * nsents
                    bbatch = {
                        'id': batch['id'][start:end],
                        'net_input': {
                            'prev_output_tokens': batch['net_input']['prev_output_tokens'][start:end],
                            'src_lengths': batch['net_input']['src_lengths'][start:end],
                            'src_tokens': batch['net_input']['src_tokens'][start:end],
                        },
                        'nsentences': batch['nsentences'] / num_splits,
                        'ntokens': batch['ntokens'] / num_splits,
                        'target': batch['target'][start:end],
                    }
                    if len(batch['id'][start:end]): # last one could be empty
                        ret.append((lang, bbatch))
        return sorted(ret, key=lambda x: x[0])


    def norm_grad(self, sample, model, criterion, optimizer, oracle_grad, ignore_grad=False):
        if ignore_grad: # if ignore_grad, run dummy_batch
            sample = self._prepare_sample(self.dummy_batch)
            loss, sample_size, logging_output = criterion(model, sample)
            loss.zero_()
            optimizer.backward(loss)
            return loss, sample_size, logging_output

        optimizer.zero_grad()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
        updated_grad = torch.zeros_like(oracle_grad[0]).cuda()
        sample = self.split_sample(sample)
        for i, (tgt_lang, tgt_lang_batch) in enumerate(sample):
            def maybe_no_sync():
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(model, "no_sync")
                    and i < len(sample) - 1
                ):
                    return model.no_sync()
                else:
                    return contextlib.nullcontext()

            with maybe_no_sync(): # only sync at the last mini-batch
                tgt_lang_id = self.lang2id[tgt_lang]
                loss, sample_size, logging_output = criterion(model, tgt_lang_batch)
                optimizer.backward(loss)
                agg_loss += loss.detach().item()
                del loss
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                if i == len(sample) - 1 or tgt_lang != sample[i+1][0]: # don't accumulate grads
                    updated_grad += self.retrieve_and_norm_grad(model, optimizer, oracle_grad, tgt_lang_id)

        return updated_grad, agg_loss, agg_sample_size, agg_logging_output


    def retrieve_and_norm_grad(self, model, optimizer, oracle_grad, lang_id):
        orcl_grad = oracle_grad[lang_id].cuda()
        if self.tgp_scope == 'model':
            train_grad = self._flatten_grad(model, optimizer)
            dot = torch.dot(train_grad, orcl_grad)
            if dot < 0:
                train_grad -= pcgrad_proj(train_grad, orcl_grad)

        elif self.tgp_scope == 'layer':
            train_grad = []
            layer_to_grads = defaultdict(list)
            for name, weight in model.named_parameters():
                name = name.split('.')
                for i, sub_name in enumerate(name):
                    if sub_name.isdigit():
                        key = '.'.join(name[:i+1])
                        break
                    if sub_name in ['weight', 'bias']:
                        key = '.'.join(name[:i])
                        break
                else:
                    raise Exception(f"Couldn't find layer for {name}")

                if weight.grad is None:
                    layer_to_grads[key].append(torch.zeros_like(weight.view(-1)))
                else:
                    layer_to_grads[key].append(weight.grad.detach().flatten())

            idx = 0
            for layer, layer_train_grad in layer_to_grads.items():
                layer_train_grad = torch.cat(layer_train_grad)
                layer_orcl_grad = orcl_grad[idx:idx + len(layer_train_grad)]
                dot = dot_product(layer_train_grad, layer_orcl_grad)
                if dot < 0:
                    layer_train_grad -= pcgrad_proj(layer_train_grad, layer_orcl_grad)

                train_grad.append(layer_train_grad)
                idx += len(layer_train_grad)

            assert idx == len(orcl_grad)
            optimizer.zero_grad()
            train_grad = torch.cat(train_grad)

        elif self.tgp_scope == 'module':
            train_grad = []
            idx = 0
            for name, weight in model.named_parameters():
                if weight.grad is None:
                    module_train_grad = torch.zeros_like(weight.view(-1))
                else:
                    module_train_grad = weight.grad.detach().flatten()

                module_orcl_grad = orcl_grad[idx: idx + len(module_train_grad)]
                dot = dot_product(module_train_grad, module_orcl_grad)
                if dot < 0:
                    module_train_grad -= pcgrad_proj(module_train_grad, module_orcl_grad)

                train_grad.append(module_train_grad)
                idx += len(module_train_grad)

            assert idx == len(orcl_grad)
            optimizer.zero_grad()
            train_grad = torch.cat(train_grad)

        return train_grad


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        oracle_grad = self.oracle_grad(update_num, model, criterion, optimizer)
        model.train()
        model.set_num_updates(update_num)
        if ignore_grad:
            return self.norm_grad(sample, model, criterion, optimizer, oracle_grad, ignore_grad=True)
        accum_grads = self._flatten_grad(model, optimizer)
        grad, loss, sample_size, logging_output = self.norm_grad(sample, model, criterion, optimizer, oracle_grad)
        self.set_grad(model, optimizer, grad + accum_grads)
        return loss, sample_size, logging_output

