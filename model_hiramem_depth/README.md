# HiRaMem future-depth auxiliary task

Sequence in the LLM:

`[BOS, current RGB + recurrent memory, prompt, depth queries × 64, action readout, autoregressive action tokens, EOS]`

The readout is one learned embedding used to predict the first action token.
It is not an action-plan query group. All tokens use ordinary causal attention:
depth queries see the prompt/current observation and earlier depth queries;
action tokens see all 64 depth queries and earlier actions. There is no custom
bidirectional attention mask. The prompt-conditioned memory reader excludes
teacher-forced action labels and padding, preventing a second leakage path.
The existing eight-component autoregressive action vocabulary is unchanged.

FlashAttention 2 remains selectable with `--attn_implementation flash_attention_2`.
Prefill computes all queries once; subsequent action tokens use the enlarged KV
cache. Generation is restricted to one episode and one beam, as in the memory
policy. Inference needs current RGBD/pose only, never future depth labels. The
head is skipped in normal generation; use `return_depth=True` on a direct
observation forward to inspect its log-depth prediction.

## Head and loss

`[B,64,D] → LayerNorm → Linear(D,256) → [B,256,8,8] → bilinear resize(16,16) → Conv3×3(256,64) → GroupNorm(8) → SiLU → Conv1×1(64,1)`

The only upsampling is 8×8 to 16×16. Queries have fixed 8×8 sine/cosine position
anchors. The head predicts log depth in metres; `exp(depth_log)` gives depth.
Ground truth and its valid mask are resized together using nearest-neighbour
sampling. The default valid range is 0.1–100 m, and pixels nearer than 10 m get
weight 3 (others get weight 1).

`Ldepth = sum(mask * weight * Huber(pred_log - log(gt))) / (sum(mask * weight) + eps) + 0.1 * Ledge`

`Ledge` is mean absolute error of horizontal/vertical adjacent log-depth
differences, including an edge only when both endpoints are valid. It uses its
own denominator. Huber delta defaults to 1. The total objective is
`Laction + depth_loss_weight * Ldepth`, with `depth_loss_weight=0.1` by default.
All loss arithmetic is FP32. Empty depth masks produce a graph-connected zero.
Separate pixel/edge/action denominators are summed across distributed ranks;
validation aggregates each numerator/denominator over the complete split.

## Labels and data alignment

GT is the next raw observation in the same episode. It is only a loss target,
never an input to memory, Q-Former, or LLM. The legacy final two STOP samples
have no future-depth supervision. Episodes with unequal frame/action lengths
keep action training but receive no depth supervision. For equal-length
episodes, raw body poses/yaws must agree with the recorded next transition
(position tolerance 0.25 m, yaw tolerance 2 degrees). This filters the source
dataset's occasional yaw snaps and incorrectly aligned transitions. The source
collection moves 3 m for forward/up/down and turns 30 degrees; these checks use
the collection's motion, not the legacy normalized action-vector magnitudes.
Missing future files and all-invalid depth maps are masked. Metrics report the
valid fraction and local cumulative filtering reasons.

## Training and checkpoints

From the project root:

```bash
python train.py
python train.py --depth_loss_weight 0.1 --depth_edge_weight 0.1
```

This project includes only `model_hiramem_depth`. Queries and the depth head
train as full auxiliary parameters outside LoRA and are saved in `future_aux.bin`.
Checkpoints include the model code, class name, configuration, and auxiliary
weights. The merger loads these saved weights and writes a full checkpoint.
See the root README for runtime paths and the 23.2% evaluation checkpoint.
