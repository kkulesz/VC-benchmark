# VC-benchmark

Models used:
- StarGANv2-VC - https://github.com/yl4579/StarGANv2-VC / https://arxiv.org/abs/2107.10394
- FreeVC - https://github.com/OlaWod/FreeVC / https://arxiv.org/abs/2210.15418


# Quick note
For both - me developing and for the future reader

## StarGANv2-VC
### start training
Given you are in project root
> source ../mgr-venv/bin/activate
>
> cd StarGANv2-VC
> 
> nohup python3 train.py --config_path ./Configs/config.yml &

### most important:
- change `model_params.num_domains` to the number of speakers in the dataset!
- data needs to be prepared. Example: https://github.com/yl4579/StarGANv2-VC/blob/main/Data/VCTK.ipynb
- data preprocessing (at least VCTK) works only under linux


## FreeVC