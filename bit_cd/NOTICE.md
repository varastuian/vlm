# Vendored code notice

The contents of `models/` and `checkpoints/BIT_LEVIR/` in this folder are
copied from the official BIT_CD implementation:

  Repository: https://github.com/justchenhao/BIT_CD
  Paper: Chen, H., Qi, Z., & Shi, Z. (2021). "Remote Sensing Image Change
  Detection with Transformers." IEEE TGRS.
  https://arxiv.org/abs/2103.00208

Only the files needed for inference (model architecture + the pretrained
LEVIR-CD checkpoint) are vendored here — not the training/eval scripts,
which are not needed by this project. No explicit open-source license file
is published in the upstream repo; these files are included here for
non-commercial research/interoperability use consistent with the authors'
publication of their code and pretrained weights for research use.
If you plan to redistribute this project publicly, check with the BIT_CD
authors regarding license terms, or replace this folder with a `git clone`
/ submodule of the upstream repo instead.
