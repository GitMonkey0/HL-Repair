# HL-Repair model card

## Model

`checkpoints/hl_repair_stb.pt` is the seed-2801 checkpoint used for the
paper's frozen-Wan endpoint. It accepts normalized local bone directions with
shape `[batch, 17, 20, 3]` and returns repaired directions of the same shape.
The training model has 1,937,300 parameters; the 100,360-parameter symbol head
is training-only, leaving 1,836,940 deployment parameters.

## Training data

The model was trained on public STB 3-D annotations from sequences B2--B6.
It uses no RGB images, identity labels, text, or generated video during
training. B1Counting and B1Random are held out for evaluation.

## Intended use

The checkpoint is intended to repair short hand-control trajectories before a
video generator or other downstream animation system. It is a research model,
not a general hand tracker or medical motion-analysis tool.

## Limitations

The model assumes a reliable palm coordinate frame and calibrated bone
lengths, uses bidirectional context, and was evaluated on one-subject STB
motion. It is not streaming and has not been validated for sign-language
recognition, multi-person interaction, object contact, or safety-critical use.

## Integrity

```text
ce93d872aebd4a06be77a7fb720a083acc24f6f955cd5a1b4e01b63e21bd5ef5  checkpoints/hl_repair_stb.pt
```
