# VC-benchmark

Models used:
- StarGANv2-VC - https://github.com/yl4579/StarGANv2-VC / https://arxiv.org/abs/2107.10394
  - works on demo deta!!!
- CVC -https://github.com/Tinglok/CVC / https://arxiv.org/abs/2011.00782
  - waiting: python 3.6 needed to be installed on titan
- TriAAN-VC - https://github.com/winddori2002/TriAAN-VC / https://arxiv.org/abs/2303.09057
- DYGAN - https://github.com/MingjieChen/DYGANVC / https://arxiv.org/abs/2203.17172

Other repositories used:
- mcd - https://github.com/MattShannon/mcd

---

# Quick note
For both - me developing and for the future reader

## StarGANv2-VC
### start training
Given you are in project root
> source ../venv-stargan-3.10/bin/activate
>
> cd StarGANv2-VC
> 
> nohup python3 train.py --config_path ./Configs/config.yml &

### most important:
- preprocessing
  - check out `preprocessing/`
  - in input dir each speaker has to be in separate directory
  - output dir has the same dir structure, with additional .txt files pointing on test/train utterances
  - data preprocessing (at least VCTK) works only under linux
- training
  - change `model_params.num_domains` to the number of speakers in the dataset!
  - run `train.py` with proper config file
- conversion/inference
  - run `inference.py` with proper paths


## TriANN-VC
### start training
Given you are in project root
> source ../venv-triann-3.10/bin/activate
>
> cd TriAAN-VC
> 
> nohup python3 main.py train &


- preprocessing
  - `preprocess.py` and `preprocess_cpc.py`
  - REMEMBER TO SET PROPER `base.yaml` and `preprocess.yaml` paths in each file you run!!
