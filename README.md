
## Preparing dataset
1. prepare dataset folder to include separate folder for each speaker inside
2. run proper `StarGANv2-VC/preprocessing` script in order to make recordings 5s each
3. run `scripts/datasets/test-train-split/split.py` to separate seen and unseen speakers and test/train
4. preprocess `TRAIN` split for model you want to use 

## StarGANv2-VC
> source ../venv-stargan-3.10/bin/activate
>
> cd StarGANv2-VC
> 
> nohup python3 -u train.py --config_path ./Configs/base-batch-size=8.yaml &
- preprocessing
  - check out `preprocessing/`
  - in input dir each speaker has to be in separate directory
  - output dir has the same dir structure, with additional .txt files pointing on test/train utterances
  - data preprocessing (at least VCTK) works only under linux
- training
  - change `model_params.num_domains` to the number of speakers in the dataset!



## TriANN-VC
> source ../venv-triann-3.10/bin/activate
>
> cd TriAAN-VC
> 
> nohup python3 -u main.py train &
- preprocessing
  - `preprocess.py` and `preprocess_cpc.py`
  - REMEMBER TO SET PROPER `base.yaml` and `preprocess.yaml` paths in each file you run!!


## Acknowledgements
- Models:
    - StarGANv2-VC - https://github.com/yl4579/StarGANv2-VC / https://arxiv.org/abs/2107.10394
    - FreeVC - https://github.com/OlaWod/FreeVC / https://arxiv.org/abs/2210.15418
    - TriAAN-VC - https://github.com/winddori2002/TriAAN-VC / https://arxiv.org/abs/2303.09057
    - DYGAN - https://github.com/MingjieChen/DYGANVC / https://arxiv.org/abs/2203.17172
    - Diff-HierVC - https://github.com/hayeong0/Diff-HierVC / https://arxiv.org/abs/2311.04693

- Other:
    - MOSNet - https://github.com/lochenchou/MOSNet / https://arxiv.org/abs/1904.08352
    - metrics - https://github.com/SandyPanda-MLDL/-Evaluation-Metrics-Used-For-The-Performance-Evaluation-of-Voice-Conversion-VC-Models
    - models propositions - https://github.com/JeffC0628/awesome-voice-conversion
---