from metrics.mcd_own_implementation.mcd import calculate_mcd


def main():
    f1 = "../FreeVC/demo_data/p225/p225_001.wav"
    f2 = "../FreeVC/demo_data/p226/p226_002.wav"

    print(f"Between the same file {calculate_mcd(f1, f1)}")
    print(f"Between different files {calculate_mcd(f1, f2)}")


if __name__ == '__main__':
    main()
