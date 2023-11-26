import torch


def main():
    print(torch.cuda.is_available())
    print(torch.__version__)


if __name__ == '__main__':
    main()
