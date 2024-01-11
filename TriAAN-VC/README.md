# TriAAN-VC: Triple Adaptive Attention Normalization for any-to-any Voice Conversion (ICASSP 2023 Top 3% paper recognized)

This is a Pytorch implementation of [TriAAN-VC: Triple Adaptive Attention Normalization for any-to-any Voice Conversion](https://arxiv.org/abs/2303.09057). TriAAN-VC is a deep learning model for any-to-any voice conversion. TriAAN-VC can maintain the linguistic contents of source speech and represent target characteristics, unlike previous methods. Experimental results on the VCTK dataset suggest that TriAAN-VC achieves state-of-the-art performance. 

We recommend you visit our [demo site](https://winddori2002.github.io/vc-demo.github.io/).

The overall architecture of TriAAN-VC is as below:

<p align="center">
	<img src="./img/triaan_vc.png" alt="TriAAN-VC" width="90%" height="90%"/>
</p>

# Installation & Enviornment

The OS, Python, and PyTorch version are as below (You can also use other versions.):
- Windows
- Linux
- python == 3.8
- pytorch == 1.9.1
- torchaudio == 0.9.1

You can install requirements through git and requirements.txt except for pytorch and torchaudio.
```C
git clone https://github.com/winddori2002/TriAAN-VC.git
cd TriAAN-VC
pip install -r requirements.txt
```

# Prepare for usage

## 1. Prepare dataset

We use the VCTK dataset consisting of 110 speakers with 400 utterances per speaker.

- The dataset can be downloaded [here](https://datashare.ed.ac.uk/handle/10283/3443).
- We divide the dataset depending on seen-to-seen and unseen-to-unseen scenarios for evaluation.

## 2. Prepare pre-trained vocoder and feature extractor

We use the pre-trained ParallelWaveGAN as vocoder and CPC extractor as feature extractor. 
You can use the [pre-trained weights in this repository](https://github.com/winddori2002/TriAAN-VC/releases/tag/v1.0). 
The vocoder is trained on the VCTK dataset and CPC extractor is trained on the LibriSpeech dataset.

- This repository provides pre-trained ParallelWaveGAN provided by [here](https://github.com/Wendison/VQMIVC) and CPC extractor provided by [here](https://github.com/facebookresearch/CPC_audio).
- Or you can train [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) and [CPC](https://github.com/facebookresearch/CPC_audio).

## 3. Preprocess

The preprocess stages contain dataset split, feature extraction, making data paths, and eval pairs.

The steps are for VCTK dataset and if you want to use other dataset, you need to modify the details.

- To split dataset and get mel-spectrograms, lf0, and metadata, run the following code and modify the directories in the ```./config/preprocess.yaml```.

```
python preprocess.py
```

- To get CPC features and represent the paths on the metadata, run the following code. (You need pre-trained weights for CPC)
```
python preprocess_cpc.py
```

- To get evaluation pairs for conversion, run the following code.
```
python generate_eval_pair.py
```

The dataset split and eval pairs are for evaluation and investigation, they are not actually necessary to train models.


# How to use

## 1. Train

### Training with settings

You can train TriAAN-VC by running the following code.

If you want to edit model settings, you can run ```python main.py train``` with other arguments. 

In ```config/base.yaml```, you can find other arguments, such as batch size, epoch, and so on.

```
python main.py train
Model arguments:
  encoder:
  	c_in:      256  (cpc:256, mel:80)
	c_h:       512
	c_out:     4
	num_layer: 6
  decoder:
  	c_in:      4
	c_h:       512
	c_out:     80
	num_layer: 6
	
Train arguments:
  epoch:      500
  batch_size: 64
  siam:       True (if not, siamese path is excluded)
  cpc:        True (if not, TriAAN-VC uses mel inputs)
```

### Training with logging

The logs are uploaded on [neptune.ai](https://neptune.ai/)
```
python main.py train --logging True

Logging arguments:
  --logging    : True/False
```

## 2. Evaluation

After training, you can evaluate the model in terms of lingustic content (WER and CER) and target characteristic (SV).

You need to keep the model arguments in the training phase. The code only supports the version in which the number of target utterances is 1.
```
python main.py test
evaluation arguments:
  --checkpoint: Checkpoint path
```

Or, you can use the below code for testing with multi-target utterances.
```
python test.py --n_uttr 1 --eval True
evaluation arguments:
  --eval:       Option for evaluation
  --n_uttr:     Number of target utterances
  --checkpoint: Checkpoint path
```

## 3. Pretrained weights

The pretrained weights of TriAAN-VC is uploaded on the github release [here](https://github.com/winddori2002/TriAAN-VC/releases/tag/v1.0).

We provide two versions of models depending on input types (mel, cpc). 

## 4. Custom convert

For custom conversion, you can run the code with ```convert.py```. 
The codes include data processing, predicting, and vocoding.

```
python convert.py 
Conversion arguments:
  --src_name:   Sample source names
  --trg_name:   Sample target names
  --checkpoint: Checkpoint path
```
You can find converted examples in ```./samples``` or please visit our [demo site](https://winddori2002.github.io/vc-demo.github.io/).

# Experimental Results

The experimental results are from the provided pre-trained weights, and the results can be slightly different from the paper.
"VCTK Split" indicates the pre-trained weight with dataset split as in the paper.

Below, the results are summarized with the "TriAAN-VC-Split". Each score is the average of seen-to-seen and unseen-to-unseen scenarios.

|Model|Pre-trained Ver.|\# uttr|WER AVG (\%)|CER AVG (\%)|SV AVG (\%)|
|:---|:---:|:---:|:---:|:---:|:---:|
|TriAAN-VC-Mel|VCTK Split|1|27.61|14.78|89.42|
|TriAAN-VC-Mel|VCTK Split|3|22.86|12.15|95.92|
|TriAAN-VC-CPC|VCTK Split|1|21.50|11.24|92.33|
|TriAAN-VC-CPC|VCTK Split|3|17.42|8.86|97.75|

<!-- We divide versions into "VCTK Split" and "VCTK All". "VCTK Split" indicates the pre-trained weight with dataset split as in the paper.
"VCTK All" means the pre-trained weight using all VCTK dataset. \# uttr indicates the number of target utterances used for conversion.

We also provide the pre-trained version of "TriAAN-VC-All". The models were trained on the whole VCTK dataset regardless of split setting.

When you use the "TriAAN-VC-All" versions, it is necessary to use "all_mel_stats.npy"

Below, the results are summarized with the "TriAAN-VC-All". The model was trained on the whole VCTK dataset regardless of split setting.

|Model|Pre-trained Ver.|\# uttr|WER AVG (\%)|CER AVG (\%)|SV AVG (\%)|
|:---|:---:|:---:|:---:|:---:|:---:|
|TriAAN-VC-Mel|VCTK All|1|27.73|15.32|90.75|
|TriAAN-VC-Mel|VCTK All|3|22.67|12.08|97.17|
|TriAAN-VC-CPC|VCTK All|1|20.72|10.62|92.92|
|TriAAN-VC-CPC|VCTK All|3|16.99|8.55|98.34| -->

## Certificate

Fortunately, our paper was recognized as the top 3% paper in ICASSP 2023.

## Citation

```
@article{park2023triaan,
  title={TriAAN-VC: Triple Adaptive Attention Normalization for Any-to-Any Voice Conversion},
  author={Park, Hyun Joon and Yang, Seok Woo and Kim, Jin Sob and Shin, Wooseok and Han, Sung Won},
  journal={arXiv preprint arXiv:2303.09057},
  year={2023}
}
```

## License

This repository is released under the MIT license. We adapted CPC codes and weights from [facebookresearch/CPC_audio](https://github.com/facebookresearch/CPC_audio), released under the MIT license. We used vocoder codes from [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), released under the MIT license. For the pre-trained vocoder, we used the weights from [Wendison/VQMIVC](https://github.com/Wendison/VQMIVCtomokui) which is released under the MIT license. We also modified preprocess codes from [Wendison/VQMIVC](https://github.com/Wendison/VQMIVCtomokui).
