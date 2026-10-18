import os
import glob
import shutil
import warnings
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

try:
    from gradescope_utils.autograder_utils.decorators import weight
except:
    # Decorator which does nothing
    def weight(n):
        return lambda func: lambda *args, **kwargs: func(*args, **kwargs)


def delayed_imports():
    """Perform imports which depend on submitted files."""
    global train_qformer
    global dit
    global image_caption_data
    import train_qformer
    import dit
    import image_caption_data


if not os.path.exists('/autograder/submission') and 'SUBDIR' not in os.environ:
    delayed_imports()


@weight(1)
def test_01_submitted_files():
    required_files = ['train_qformer.py', 'dit.py', 'image_caption_data.py']
    submission_dir = '/autograder/submission'
    if 'SUBDIR' in os.environ:
        submission_dir = os.environ['SUBDIR']
    if os.path.exists(submission_dir):
        # We are running on Gradescope
        print('Submitted files: ', end='')
        print([x.replace(submission_dir, '') for x in glob.glob(f'{submission_dir}/**/*', recursive=True)])
        # Copy required files to current directory for import, and track any missing files
        missing_files = []
        for file_name in required_files:
            src = f'{submission_dir}/{file_name}'
            if os.path.exists(src):
                shutil.copy(src, f'./{file_name}')
            else:
                missing_files.append(file_name)
        assert len(missing_files) == 0, f"Missing files: {missing_files}"
    delayed_imports()


#####################################################################
# Part 1: tests for prompts_to_padded_hidden_states
#####################################################################


class DummyBatch(dict):
    def to(self, device: torch.device):
        self["device"] = device
        return self


class DummyTokenizer:
    def __init__(self, vocab: Dict[str, int], call_log: List[Tuple[str, int]]):
        self.vocab = vocab
        self.call_log = call_log
        self._counter = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> DummyBatch:
        assert return_tensors == "pt"
        # Record tokenizer call with prompt index
        prompt_index = self._counter
        self._counter += 1
        self.call_log.append(("tokenizer", prompt_index))

        tokens = [self.vocab[w] for w in text.strip().split()] if text.strip() else []
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, S)
        batch = DummyBatch({"input_ids": input_ids, "prompt_index": prompt_index})
        return batch


class DummyOutputs:
    def __init__(self, hidden_states: Tuple[torch.Tensor, ...]):
        self.hidden_states = hidden_states


class DummyGPT2:
    def __init__(self, num_layers: int, hidden_size: int, call_log: List[Tuple[str, int]]):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.call_log = call_log

    def __call__(self, *, input_ids: torch.Tensor, output_hidden_states: bool, prompt_index: int, **kwargs):
        assert output_hidden_states is True
        # Record gpt2 call after tokenizer for the same prompt index
        self.call_log.append(("gpt2", prompt_index))

        # Build per-layer simple hidden states: repeat token ids across channel dim and offset by layer idx
        # Shapes: (1, S, C)
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, self.hidden_size)
        layers = tuple(base + float(layer_idx) for layer_idx in range(self.num_layers))
        return DummyOutputs(hidden_states=layers)


@weight(1)
def test_11_layer_index_selection():
    call_log: List[Tuple[str, int]] = []
    vocab = {"hello": 1, "world": 2, "a": 3}
    tokenizer = DummyTokenizer(vocab, call_log)
    gpt2 = DummyGPT2(num_layers=3, hidden_size=4, call_log=call_log)
    device = torch.device("cpu")

    prompts = ["hello world", "a"]
    idx = 1
    output, masks = image_caption_data.prompts_to_padded_hidden_states(
        prompts=prompts,
        gpt2=gpt2,
        tokenizer=tokenizer,
        gpt2_layer_index=idx,
        device=device,
    )

    # Shape checks
    assert output.shape[0] == 2  # B
    assert output.shape[1] == 2  # max_seq ("hello world" has len 2)
    assert output.shape[2] == 4  # C

    # Content check: equals token id + idx for actual token positions
    # Prompt 0 tokens: [1, 2]
    expected0 = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]) + idx
    torch.testing.assert_close(output[0, :2], expected0)

    # Prompt 1 tokens: [3] then one pad row
    expected1_token = torch.tensor([[3.0, 3.0, 3.0, 3.0]]) + idx
    torch.testing.assert_close(output[1, :1], expected1_token)


@weight(1)
def test_12_tokenizer_called_before_gpt2():
    call_log: List[Tuple[str, int]] = []
    vocab = {"x": 5, "y": 6}
    tokenizer = DummyTokenizer(vocab, call_log)
    gpt2 = DummyGPT2(num_layers=2, hidden_size=3, call_log=call_log)
    device = torch.device("cpu")

    prompts = ["x", "x y"]
    _output, _masks = image_caption_data.prompts_to_padded_hidden_states(
        prompts=prompts,
        gpt2=gpt2,
        tokenizer=tokenizer,
        gpt2_layer_index=0,
        device=device,
    )

    # For each prompt index, tokenizer event must precede gpt2 event
    events_by_prompt: Dict[int, List[str]] = {}
    for kind, prompt_index in call_log:
        events_by_prompt.setdefault(prompt_index, []).append(kind)

    for prompt_index, events in events_by_prompt.items():
        assert len(events) >= 2
        assert events[0] == "tokenizer"
        assert events[1] == "gpt2"


@weight(1)
def test_13_padding_correct():
    call_log: List[Tuple[str, int]] = []
    vocab = {"one": 1, "two": 2, "three": 3}
    tokenizer = DummyTokenizer(vocab, call_log)
    gpt2 = DummyGPT2(num_layers=2, hidden_size=2, call_log=call_log)
    device = torch.device("cpu")

    prompts = ["one two three", "one"]
    output, masks = image_caption_data.prompts_to_padded_hidden_states(
        prompts=prompts,
        gpt2=gpt2,
        tokenizer=tokenizer,
        gpt2_layer_index=0,
        device=device,
    )

    # Second prompt length is 1 -> positions 1 and 2 should be zeros
    assert torch.all(output[1, 1:].eq(0))


@weight(1)
def test_14_invalid_index_below_zero():
    call_log: List[Tuple[str, int]] = []
    vocab = {"z": 9}
    tokenizer = DummyTokenizer(vocab, call_log)
    gpt2 = DummyGPT2(num_layers=2, hidden_size=2, call_log=call_log)
    device = torch.device("cpu")

    with pytest.raises(ValueError):
        _ = image_caption_data.prompts_to_padded_hidden_states(
            prompts=["z"],
            gpt2=gpt2,
            tokenizer=tokenizer,
            gpt2_layer_index=-1,
            device=device,
        )


@weight(1)
def test_15_invalid_index_above_range():
    call_log: List[Tuple[str, int]] = []
    vocab = {"z": 9}
    tokenizer = DummyTokenizer(vocab, call_log)
    gpt2 = DummyGPT2(num_layers=2, hidden_size=2, call_log=call_log)
    device = torch.device("cpu")

    with pytest.raises(ValueError):
        _ = image_caption_data.prompts_to_padded_hidden_states(
            prompts=["z"],
            gpt2=gpt2,
            tokenizer=tokenizer,
            gpt2_layer_index=2,  # equal to L -> out of range
            device=device,
        )


@weight(1)
def test_16_masks_correct():
    call_log: List[Tuple[str, int]] = []
    # Include a negative token id to ensure some GPT2 features are negative
    vocab = {"one": 1, "two": -2, "three": 3}
    tokenizer = DummyTokenizer(vocab, call_log)
    gpt2 = DummyGPT2(num_layers=2, hidden_size=2, call_log=call_log)
    device = torch.device("cpu")

    prompts = ["one two three", "one"]
    _output, masks = image_caption_data.prompts_to_padded_hidden_states(
        prompts=prompts,
        gpt2=gpt2,
        tokenizer=tokenizer,
        gpt2_layer_index=0,
        device=device,
    )

    # Masks should be boolean with True for tokens and False for pads
    assert masks.dtype == torch.bool
    assert tuple(masks.shape) == (2, 3)
    expected0 = torch.tensor([True, True, True])
    expected1 = torch.tensor([True, False, False])
    assert torch.equal(masks[0], expected0)
    assert torch.equal(masks[1], expected1)


#####################################################################
# Part 2: tests for setup_optimizer_and_scheduler
#####################################################################


class DummyQueryEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 4)
        self.linear2 = nn.Linear(4, 2)

    def named_parameters(self, prefix: str = '', recurse: bool = True):  # type: ignore[override]
        for name, p in super().named_parameters(prefix=prefix, recurse=recurse):
            yield name, p


def _build_real_model() -> nn.Module:
    # Construct a small instance with text conditioning enabled
    model = dit.DiT_Llama(
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=64,
        n_layers=2,
        n_heads=4,
        num_classes=10,
        use_text_conditioning=True,
        transformer_hidden_size=32,
        num_query_tokens=2,
        qe_num_layers=1,
        qe_n_heads=2,
    )
    return model


class Args:
    def __init__(self):
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 50


def _build_model_and_args():
    model = _build_real_model()
    # Mark all params requires_grad=False initially to mimic train_qformer behavior
    for p in model.parameters():
        p.requires_grad = False
    args = Args()
    return model, args


@weight(1)
def test_21_only_query_embedder_params_have_gradients():
    model, args = _build_model_and_args()
    optimizer, scheduler, criterion = train_qformer.setup_optimizer_and_scheduler(model, args)

    # Non-query params should not require grad; query_embedder params should
    for name, p in model.named_parameters():
        if name.startswith('query_embedder.'):
            assert p.requires_grad, f"Query embedder param {name} should be trainable"
        else:
            assert not p.requires_grad, f"Non-query param {name} should be frozen"

    # Optimizer should only include trainable params
    opt_params = list(optimizer.param_groups[0]['params'])
    trainables = [p for p in model.parameters() if p.requires_grad]
    assert set(opt_params) == set(trainables)


@weight(1)
def test_23_adamw_arguments():
    model, args = _build_model_and_args()
    optimizer, scheduler, criterion = train_qformer.setup_optimizer_and_scheduler(model, args)

    assert isinstance(optimizer, torch.optim.AdamW)
    # Verify key hyperparameters
    # Use defaults/base LR to avoid version-dependent scheduler init behavior
    assert optimizer.defaults.get('lr') == pytest.approx(args.lr)
    if scheduler is not None:
        assert scheduler.base_lrs[0] == pytest.approx(args.lr)
    assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(args.weight_decay)
    assert optimizer.defaults.get('eps') == args.adam_epsilon
    # betas are stored in param_groups
    assert optimizer.param_groups[0]['betas'] == (0.9, 0.95)


@weight(1)
def test_24_scheduler_setup():
    model, args = _build_model_and_args()
    optimizer, scheduler, criterion = train_qformer.setup_optimizer_and_scheduler(model, args)

    # With warmup_steps > 0, scheduler should not be None and be a LambdaLR
    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    # Check warmup shape: at step 0 lr scaled by 1/warmup_steps based on base_lrs
    base_lr = scheduler.base_lrs[0]
    scheduler.last_epoch = -1  # before any step

    with warnings.catch_warnings():
        # Filter warning about calling scheduler.step() before optimizer.step()
        warnings.simplefilter("ignore")
        scheduler.step()  # step 0
    expected_scale = min(1.0, float(0 + 1) / float(args.warmup_steps))
    assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr * expected_scale)


@weight(1)
def test_25_criterion_setup():
    model, args = _build_model_and_args()
    optimizer, scheduler, criterion = train_qformer.setup_optimizer_and_scheduler(model, args)
    assert isinstance(criterion, nn.MSELoss)


#####################################################################
# Part 3: tests of QueryEmbedder.forward()
#####################################################################


@weight(1)
def test_31_output_shape():
    """Test that QueryEmbedder.forward() produces correct output shape"""
    torch.manual_seed(42)

    # Create a small QueryEmbedder instance
    transformer_hidden_size = 16
    conditioning_hidden_size = 12
    num_queries = 4
    num_dit_layers = 3

    query_embedder = dit.QueryEmbedder(
        transformer_hidden_size=transformer_hidden_size,
        conditioning_hidden_size=conditioning_hidden_size,
        num_queries=num_queries,
        num_dit_layers=num_dit_layers,
        num_layers=1,
        n_heads=2,
    )
    query_embedder.eval()

    # Create input: (batch_size=2, seq_len=5, hidden_size=16)
    batch_size = 2
    seq_len = 5
    cross_attention_states = torch.randn(batch_size, seq_len, transformer_hidden_size)

    # Forward pass
    output = query_embedder.forward(cross_attention_states, batch_size=batch_size)

    # Check output shape: (batch_size, num_dit_layers, conditioning_hidden_size)
    expected_shape = (batch_size, num_dit_layers, conditioning_hidden_size)
    assert output.shape == torch.Size(expected_shape)
    assert output.shape[0] == batch_size
    assert output.shape[1] == num_dit_layers
    assert output.shape[2] == conditioning_hidden_size


@weight(1)
def test_32_without_attention_mask():
    """Test that QueryEmbedder.forward() produces deterministic output"""
    torch.manual_seed(123)

    # Create a small QueryEmbedder instance
    transformer_hidden_size = 8
    conditioning_hidden_size = 6
    num_queries = 3
    num_dit_layers = 2

    query_embedder = dit.QueryEmbedder(
        transformer_hidden_size=transformer_hidden_size,
        conditioning_hidden_size=conditioning_hidden_size,
        num_queries=num_queries,
        num_dit_layers=num_dit_layers,
        num_layers=1,
        n_heads=2,
    )
    query_embedder.eval()

    # Create specific input (not completely random - use arange for reproducibility)
    batch_size = 2
    seq_len = 3
    cross_attention_states = torch.arange(
        batch_size * seq_len * transformer_hidden_size, dtype=torch.float32
    ).reshape(batch_size, seq_len, transformer_hidden_size) * 0.1

    # Forward pass
    with torch.no_grad():
        output = query_embedder.forward(cross_attention_states, batch_size=batch_size)

    # Expected output captured from reference implementation
    expected_output = torch.tensor(
        [
            [[-0.1994, 0.1614, -0.0371, 0.2265, 0.0294, -0.0398], [0.6387, 0.4361, 1.1713, -0.7010, 0.5208, 0.2349]],
            [[-0.1085, 0.1773, -0.0027, 0.3100, 0.0573, -0.0557], [-0.0621, 0.3293, 0.9645, -1.4534, 0.2647, 0.3710]],
        ]
    )

    # Verify output matches expected
    torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)


@weight(1)
def test_33_with_attention_mask():
    """Test QueryEmbedder.forward() with cross_attention_mask"""
    torch.manual_seed(456)

    # Create a QueryEmbedder instance with different dimensions
    transformer_hidden_size = 12
    conditioning_hidden_size = 8
    num_queries = 2
    num_dit_layers = 3

    query_embedder = dit.QueryEmbedder(
        transformer_hidden_size=transformer_hidden_size,
        conditioning_hidden_size=conditioning_hidden_size,
        num_queries=num_queries,
        num_dit_layers=num_dit_layers,
        num_layers=2,
        n_heads=4,
    )
    query_embedder.eval()

    # Create input with varying sequence lengths (simulating padded batch)
    batch_size = 2
    seq_len = 4
    # Use arange-based input for reproducibility
    cross_attention_states = torch.arange(
        batch_size * seq_len * transformer_hidden_size, dtype=torch.float32
    ).reshape(batch_size, seq_len, transformer_hidden_size) * 0.05

    # Create attention mask: first sequence has 3 valid tokens, second has 2 valid tokens
    # True = valid token, False = padding
    cross_attention_mask = torch.tensor(
        [
            [True, True, True, False],  # First sequence: 3 valid tokens, 1 padding
            [True, True, False, False],  # Second sequence: 2 valid tokens, 2 padding
        ]
    )

    # Forward pass
    with torch.no_grad():
        output = query_embedder.forward(
            cross_attention_states,
            batch_size=batch_size,
            cross_attention_mask=cross_attention_mask,
        )

    # Expected output captured from reference implementation
    expected_output = torch.tensor(
        [
            [
                [-0.1589, -0.0489, -0.1457, 0.1873, 0.4370, 0.2996, -0.1497, -0.5045],
                [0.5129, 0.2228, 0.0636, -0.5330, -0.5454, 0.6732, 0.0466, 0.4089],
                [0.1482, 0.0534, 0.0867, -0.1660, -0.3031, -0.0813, 0.0869, 0.3232],
            ],
            [
                [-0.1347, 0.2199, -0.1756, 0.0603, 0.7269, 0.1671, -0.3748, -0.2418],
                [0.8336, -0.1070, -0.0546, -0.6384, -1.1150, 1.2122, 0.2823, -0.0427],
                [0.1738, -0.1323, 0.0887, -0.1098, -0.5291, 0.0556, 0.2375, 0.1261],
            ],
        ]
    )

    # Verify output matches expected
    torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)
