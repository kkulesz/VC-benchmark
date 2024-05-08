import os
import argparse
from random import shuffle

import paths

if __name__ == "__main__":
    filelist_dir = paths.FLIST_PATH
    os.makedirs(filelist_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default=f"{filelist_dir}/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default=f"{filelist_dir}/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default=f"{filelist_dir}/test.txt", help="path to test list")
    parser.add_argument("--source_dir", type=str, default=paths.DOWNSAMPLED_16k_PATH, help="path to source dir")
    args = parser.parse_args()
    
    train = []
    val = []
    test = []
    idx = 0
    
    for speaker in os.listdir(args.source_dir):
        wavs = os.listdir(os.path.join(args.source_dir, speaker))
        shuffle(wavs)
        train += wavs[2:-10]
        val += wavs[:2]
        test += wavs[-10:]
        
    shuffle(train)
    shuffle(val)
    shuffle(test)

    with open(args.train_list, "w") as f:
        for fname in train:
            # speaker = fname[:4]
            # wavpath = os.path.join("DUMMY", speaker, fname)
            # f.write(wavpath + "\n")
            f.write(fname + "\n")

    with open(args.val_list, "w") as f:
        for fname in val:
            # speaker = fname[:4]
            # wavpath = os.path.join("DUMMY", speaker, fname)
            # f.write(wavpath + "\n")
            f.write(fname + "\n")

    with open(args.test_list, "w") as f:
        for fname in test:
            # speaker = fname[:4]
            # wavpath = os.path.join("DUMMY", speaker, fname)
            # f.write(wavpath + "\n")
            f.write(fname + "\n")
            