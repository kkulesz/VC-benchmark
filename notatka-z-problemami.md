28.11.2023
- StarGAN:
  - `TitanV` - `cpu` - `batch_size=1`: IndexError: tuple index out of range. Prawdopodobnie coś  jest zepsute w handlowaniu wymiarów.
  - `TitanV` - `cpu` - `batch_size=2`: iteracja trwa ~90sekund. Jedna epoka składa się z 1362 iteracji. W konfigu wstępnie było 150 epoch. Trwałoby to okolo 190 dni
  - `TitanV` - `cuda` - OOM, w różnych konfiguracjach ustawień.
  - TODO: sprawdzic u siebie ,czy mam wystarczajaco pamięci. Najlepiej na pythonie 3.7. Ale ogolnie stworzyc kilka srodowisk