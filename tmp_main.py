import torch
from tqdm import tqdm
import time


def print_machine_props():
    print(f"Torch version: {torch.__version__}")
    print(f"Is Cuda available: {torch.cuda.is_available()}")
    gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpus}")
    for i in range(gpus):
        def to_gb(bytes):
            return round(bytes / 1024 / 1024 / 1024, 3)

        t = torch.cuda.get_device_properties(i).total_memory
        r = torch.cuda.memory_reserved(i)
        a = torch.cuda.memory_allocated(i)
        print(f"\tdevice nr {i}:")
        print(f"\ttotal: \t\t{to_gb(t)} GB")
        print(f"\treserved: \t{to_gb(r)} GB")
        print(f"\tallocated: \t{to_gb(a)} GB")


def test_tqdm():
    l = range(0, 10)
    s = 0
    for i, a in enumerate(tqdm(l)):
        time.sleep(1)
        s = a
    print(s)


def test_state_dict():
    # checkpoint_path = "StarGANv2-VC/Models/TEST/final_00001_epochs.pth"
    # state_dict = torch.load(checkpoint_path, map_location="cpu")
    #
    # print(state_dict.keys())
    # print(state_dict['model_ema'].keys())
    # print(state_dict['model'].keys())
    #
    # generator = state_dict['model_ema']['generator']
    # stargan = state_dict['model_ema']
    #
    # torch.save(generator, "generator.pth")
    # torch.save(stargan, "stargan.pth")

    checkpoint_path = "StarGANv2-VC/Models/epoch_00150.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # print(state_dict.keys())

def main():
    # print_machine_props()
    # test_tqdm()
    test_state_dict()


if __name__ == '__main__':
    main()
