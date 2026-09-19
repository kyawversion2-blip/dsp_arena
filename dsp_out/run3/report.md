# DSP Inverse Optimizer - Report

- **Mode**: `generic` (focus split `AB|CD|AB`)
- **Best DSP params**: `sr=384000`, `n_fft=4096`, `hop_length=2048`, `top_db=27`, `window=0.45s`
- **Objective** (trial 11/12): `3.29229`
- **Best overall split**: `B|AD` p=0.0005 @ `duration`
- **Silhouette** (z-scored core features): -0.0465
- **Files used**: 321  (A=67 B=79 C=96 D=79)

## Feature importance (one-at-a-time param sensitivity)

| param | from -> to | objective delta | most-moved features |
|---|---|---|---|
| sr | 384000 -> 192000 | -0.9811 | impact_freq_hz, mel_mean_db, peak_time |
| n_fft | 4096 -> 8192 | +0.0022 | impact_freq_hz, mel_mean_db, impact |
| n_fft | 4096 -> 2048 | +0.0002 | impact_freq_hz, mel_mean_db, impact |
| hop_length | 2048 -> 1024 | +0.0000 | impact, duration, peak_time |
| window_sec | 0.45 -> 0.4 | +0.0000 | impact, duration, peak_time |
| hop_length | 2048 -> 4096 | +0.0000 | impact, duration, peak_time |
| window_sec | 0.45 -> 0.5 | +0.0000 | impact, duration, peak_time |
| top_db | 27 -> 26 | +0.0000 | impact, duration, peak_time |

## Mathematical relationship (parameter -> feature -> separation)

### `sr`: 384000 -> 192000
- `impact`: 0.502606 -> 0.479642 (delta -0.022965)
- `duration`: 0.657429 -> 0.665238 (delta +0.007809)
- `peak_time`: 3.599425 -> 3.90123 (delta +0.301805)
- `mel_mean_db`: 18.895154 -> 20.901236 (delta +2.006082)
- `impact_freq_hz`: 2409.882866 -> 2341.581931 (delta -68.300935)

### `n_fft`: 4096 -> 8192
- `impact`: 0.502606 -> 0.502606 (delta +0.000000)
- `duration`: 0.657429 -> 0.657429 (delta +0.000000)
- `peak_time`: 3.599425 -> 3.599425 (delta +0.000000)
- `mel_mean_db`: 18.895154 -> 26.123979 (delta +7.228825)
- `impact_freq_hz`: 2409.882866 -> 2204.474455 (delta -205.408411)

### `n_fft`: 4096 -> 2048
- `impact`: 0.502606 -> 0.502606 (delta +0.000000)
- `duration`: 0.657429 -> 0.657429 (delta +0.000000)
- `peak_time`: 3.599425 -> 3.599425 (delta +0.000000)
- `mel_mean_db`: 18.895154 -> 2.257128 (delta -16.638026)
- `impact_freq_hz`: 2409.882866 -> 2231.34891 (delta -178.533956)

### `hop_length`: 2048 -> 1024
- `impact`: 0.502606 -> 0.502606 (delta +0.000000)
- `duration`: 0.657429 -> 0.657429 (delta +0.000000)
- `peak_time`: 3.599425 -> 3.599425 (delta +0.000000)
- `mel_mean_db`: 18.895154 -> 18.895154 (delta +0.000000)
- `impact_freq_hz`: 2409.882866 -> 2409.882866 (delta +0.000000)

## 25-split p-value table (winning params)

| split | impact | duration | peak_time | mel_mean_db | impact_freq_hz | best |
|---|---|---|---|---|---|---|
| A|B | 0.2998 | 0.0016 | 0.0778 | 0.1621 | 0.0553 | 0.0016 |
| A|C | 0.3322 | 0.3777 | 0.7550 | 0.8462 | 0.1725 | 0.1725 |
| A|D | 0.3459 | 0.6529 | 0.3602 | 0.9062 | 0.8244 | 0.3459 |
| B|C | 0.8820 | 0.0178 | 0.1637 | 0.0866 | 0.5457 | 0.0178 |
| B|D | 0.7569 | 0.0043 | 0.3372 | 0.0951 | 0.1014 | 0.0043 |
| C|D | 0.9224 | 0.6867 | 0.6346 | 0.9414 | 0.2571 | 0.2571 |
| A|BC | 0.2573 | 0.0290 | 0.2660 | 0.5408 | 0.0669 | 0.0290 |
| A|BD | 0.2585 | 0.0400 | 0.1264 | 0.4660 | 0.2222 | 0.0400 |
| A|CD | 0.2785 | 0.4396 | 0.5035 | 0.8567 | 0.3436 | 0.2785 |
| B|AC | 0.6752 | 0.0018 | 0.0735 | 0.0692 | 0.1766 | 0.0018 |
| B|AD | 0.7081 | 0.0005 | 0.1196 | 0.0727 | 0.0395 | 0.0005 |
| B|CD | 0.7976 | 0.0030 | 0.1716 | 0.0523 | 0.2144 | 0.0030 |
| C|AB | 0.5264 | 0.3216 | 0.4856 | 0.2374 | 0.7039 | 0.2374 |
| C|AD | 0.6329 | 0.4567 | 0.9021 | 0.8769 | 0.1416 | 0.1416 |
| C|BD | 0.9768 | 0.2486 | 0.2721 | 0.2934 | 0.7566 | 0.2486 |
| D|AB | 0.4796 | 0.1301 | 0.9273 | 0.2734 | 0.3744 | 0.1301 |
| D|AC | 0.5929 | 0.9695 | 0.4427 | 0.9922 | 0.5296 | 0.4427 |
| D|BC | 0.8219 | 0.0779 | 0.8291 | 0.4012 | 0.1170 | 0.0779 |
| A|BCD | 0.2409 | 0.0810 | 0.2511 | 0.6922 | 0.1598 | 0.0810 |
| B|ACD | 0.8620 | 0.0008 | 0.0965 | 0.0492 | 0.1049 | 0.0008 |
| C|ABD | 0.6892 | 0.6057 | 0.4859 | 0.3900 | 0.4454 | 0.3900 |
| D|ABC | 0.6128 | 0.2405 | 0.8719 | 0.4979 | 0.2685 | 0.2405 |
| AB|CD | 0.4202 | 0.1361 | 0.6167 | 0.1684 | 0.7989 | 0.1361 |
| AC|BD | 0.5576 | 0.0585 | 0.1160 | 0.2671 | 0.6576 | 0.0585 |
| AD|BC | 0.6040 | 0.0147 | 0.4257 | 0.3628 | 0.0353 | 0.0147 |
| BC|AD | 0.6040 | 0.0147 | 0.4257 | 0.3628 | 0.0353 | 0.0147 |
| BD|AC | 0.5576 | 0.0585 | 0.1160 | 0.2671 | 0.6576 | 0.0585 |
| CD|AB | 0.4202 | 0.1361 | 0.6167 | 0.1684 | 0.7989 | 0.1361 |

## Fisher ratio per feature

- `impact`: 0.3651
- `duration`: 3.2165
- `peak_time`: 1.5876
- `mel_mean_db`: 0.1750
- `impact_freq_hz`: 1.4589

## Mahalanobis centroid distance (per feature-pair, focus split)

- `AB|CD`: impact=0.005, duration=0.056, peak_time=0.018, mel_mean_db=0.019, impact_freq_hz=0.020
- `A|D`: impact=0.039, duration=0.028, peak_time=0.057, mel_mean_db=0.047, impact_freq_hz=0.039
- `B|C`: impact=0.033, duration=0.140, peak_time=0.068, mel_mean_db=0.001, impact_freq_hz=0.005
- `A|B`: impact=0.020, duration=0.164, peak_time=0.128, mel_mean_db=0.010, impact_freq_hz=0.109
- `C|D`: impact=0.053, duration=0.002, peak_time=0.001, mel_mean_db=0.030, impact_freq_hz=0.060

> Reproduce: `python3 dsp_inverse.py reproduce --out run3`