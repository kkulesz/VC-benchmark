from metrics.mcd_own_implementation.mcd import calculate_mcd


def main():
    converted = "../../samples/TEST-TRIANN/p270/34.wav"
    trg_spk = "../../samples/TEST-TRIANN/p270/rec_34.wav"
    src_spk = "../../samples/TEST-TRIANN/source_gen.wav"

    print(f"Between the same file: {calculate_mcd(converted, converted)}")
    print(f"Between trg speaker and converted: {calculate_mcd(trg_spk, converted)}")
    print(f"Between src speaker and converted: {calculate_mcd(src_spk, converted)}")


if __name__ == '__main__':
    main()
