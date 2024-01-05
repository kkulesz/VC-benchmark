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
    l = range(0,10)
    s = 0
    for i, a in enumerate(tqdm(l)):
        time.sleep(1)
        s = a
    print(s)


def main():
    # print_machine_props()
    test_tqdm()

if __name__ == '__main__':
    main()
