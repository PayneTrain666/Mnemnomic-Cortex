from __future__ import annotations

import torch

from tools.babi_train_eval import (
    _answer_logits,
    _task_number,
    build_arg_parser,
    build_flat_examples,
    build_vocab,
    Example,
)


def test_babi_defaults_match_copy_training_profile_and_qdt_safety():
    args = build_arg_parser().parse_args([])

    assert args.d_model == 160
    assert args.batch_size == 32
    assert args.lr == 3e-4
    assert args.weight_decay == 0.02
    assert args.warmup_ratio == 0.10
    assert args.min_lr_ratio == 0.08
    assert args.target_grad_norm == 0.8
    assert args.grad_clip == 1.0
    assert args.enable_task_decoder is True
    assert args.task_decoder_layers == 4
    assert args.task_decoder_heads == 8
    assert args.use_amp is True
    assert args.working_memory_fabric == "qdt"
    assert args.qdt_hardware_profile == "single_gpu_8_12gb"
    assert args.dataset_source == "mirror"


def test_answer_logits_uses_last_non_padding_token_per_story():
    logits = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    src = torch.tensor(
        [
            [4, 5, 6, 0, 0],
            [7, 8, 9, 10, 0],
        ]
    )

    selected = _answer_logits(logits, src, pad_id=0)

    assert torch.equal(selected[0], logits[0, 2])
    assert torch.equal(selected[1], logits[1, 3])


def test_babi_vocabulary_can_exclude_official_test_split():
    train = [Example(tokens=["mary", "kitchen"], answer="kitchen")]
    test = [Example(tokens=["sandra", "garden"], answer="garden")]

    vocab = build_vocab(train, [])

    assert "mary" in vocab
    assert "kitchen" in vocab
    assert "sandra" not in vocab
    assert "garden" not in vocab


def test_mirror_rows_and_config_are_normalized():
    rows = [
        {
            "passage": "Mary moved to the bathroom.\n",
            "question": "Where is Mary?",
            "answer": "Bathroom",
            "task": 1,
        }
    ]
    examples = build_flat_examples(rows)

    assert _task_number("en-10k-qa1") == 1
    assert examples[0].answer == "bathroom"
    assert examples[0].tokens[:2] == ["<ctx>", "mary"]

