import os

FLIST_PATH = "./filelists_mine"

DATA_DIR = "../../Data"
RAW_VCTK_PATH = os.path.join(DATA_DIR, "VCTK", "wav48_silence_trimmed")

FREEVC_DATA_PATH = os.path.join(DATA_DIR, "freeVc-preprocessed")
DOWNSAMPLED_16k_PATH = os.path.join(FREEVC_DATA_PATH, "vctk-16k")
DOWNSAMPLED_22k_PATH = os.path.join(FREEVC_DATA_PATH, "vctk-22k")

SPK_PATH = os.path.join(FREEVC_DATA_PATH, "spk")
WAVLM_PATH = os.path.join(FREEVC_DATA_PATH, "wavlm")

# SR paths
SR_PATH = os.path.join(FREEVC_DATA_PATH, "sr")
SR_WAV_PATH = os.path.join(SR_PATH, "wav")
SR_SSL_PATH = os.path.join(SR_PATH, "wavlm")

