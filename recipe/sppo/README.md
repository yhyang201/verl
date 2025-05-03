# SPPO: Self-Play Preference Optimization for Language Model Alignment

This repository hosts the community implementation for the paper [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675).

Paper Authors: [Yue Wu](https://yuewu.us/)\*, [Zhiqing Sun](https://www.cs.cmu.edu/~zhiqings/)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Yiming Yang](https://www.cs.cmu.edu/~yiming/), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

veRL Implementation Authors: [Yuhao Yang](https://github.com/yhyang201), [Chenyang Zhao](https://github.com/zhaochenyang20)

[[Webpage](https://uclaml.github.io/SPPO/)] [[Huggingface](https://huggingface.co/papers/2405.00675)] [[Paper](https://arxiv.org/abs/2405.00675)][[Original Implementation](https://github.com/uclaml/SPPO)]

## Quickstart

```
cd verl
python3 -m uv pip install -e ".[sglang]"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3
bash recipe/sppo/run_qwen2.5-7b_rm.sh
```

## Acknowledgement

We sincerely thank the contribution and guidance from:

- [Yue Wu](https://yuewu.us/)
- [Chendong Wang](https://cdwang96.github.io/)
- [Yifan Zhang](https://github.com/yifanzhang-pro)
- [Yongan Xiang](https://github.com/BearBiscuit05)
- [Junrong Lin](https://github.com/ocss884)
- [Yuxuan Tong](https://github.com/tongyx361)
- [Guangming Shen](https://github.com/PeterSH6)
- [Biao He](https://www.linkedin.com/in/biao-he/)
- [Qingquan Song](https://qingquansong.github.io/)
- [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)