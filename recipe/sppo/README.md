# SPPO: Self-Play Preference Optimization for Language Model Alignment

This repository hosts the community implementation for the paper [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675).

Authors: [Yue Wu](https://yuewu.us/)\*, [Zhiqing Sun](https://www.cs.cmu.edu/~zhiqings/)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Yiming Yang](https://www.cs.cmu.edu/~yiming/), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)


[[Webpage](https://uclaml.github.io/SPPO/)] [[Huggingface](https://huggingface.co/papers/2405.00675)] [[Paper](https://arxiv.org/abs/2405.00675)]


Official Implementation:
The official implementation repository from the authors can be found here: [https://github.com/uclaml/SPPO](https://github.com/uclaml/SPPO)


## Quickstart

```
cd verl
python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct
cd recipe/sppo
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_qwen2.5-7b_rm.sh
```

