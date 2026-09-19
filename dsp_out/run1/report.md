# DSP Inverse Optimizer - Report

- **Mode**: `generic` (focus split `AB|CD|AB`)
- **Best DSP params**: `sr=384000`, `n_fft=1024`, `hop_length=4096`, `top_db=23`, `window=0.7s`
- **Objective** (trial 38/50): `3.47824`
- **Best overall split**: `B|AC` p=0.0003 @ `duration`
- **Silhouette** (z-scored core features): -0.0268
- **Files used**: 321  (A=67 B=79 C=96 D=79)

## Feature importance (one-at-a-time param sensitivity)

| param | from -> to | objective delta | most-moved features |
|---|---|---|---|
| sr | 384000 -> 192000 | -0.7657 | impact_freq_hz, mel_mean_db, peak_time |
| n_fft | 1024 -> 512 | -0.0024 | impact_freq_hz, mel_mean_db, impact |
| n_fft | 1024 -> 2048 | -0.0007 | impact_freq_hz, mel_mean_db, impact |
| hop_length | 4096 -> 2048 | +0.0000 | impact, duration, peak_time |
| window_sec | 0.7 -> 0.65 | +0.0000 | impact, duration, peak_time |
| window_sec | 0.7 -> 0.75 | +0.0000 | impact, duration, peak_time |
| top_db | 23 -> 22 | +0.0000 | impact, duration, peak_time |
| top_db | 23 -> 24 | +0.0000 | impact, duration, peak_time |

## Mathematical relationship (parameter -> feature -> separation)

### `sr`: 384000 -> 192000
- `impact`: 0.503063 -> 0.479631 (delta -0.023432)
- `duration`: 0.735394 -> 0.742613 (delta +0.007219)
- `peak_time`: 4.608556 -> 4.775823 (delta +0.167267)
- `mel_mean_db`: -32.158662 -> -3.033277 (delta +29.125386)
- `impact_freq_hz`: 1701.023364 -> 1321.673209 (delta -379.350156)

### `n_fft`: 1024 -> 512
- `impact`: 0.503063 -> 0.503063 (delta +0.000000)
- `duration`: 0.735394 -> 0.735394 (delta +0.000000)
- `peak_time`: 4.608556 -> 4.608556 (delta +0.000000)
- `mel_mean_db`: -32.158662 -> -76.28208 (delta -44.123417)
- `impact_freq_hz`: 1701.023364 -> 1768.356075 (delta +67.332710)

### `n_fft`: 1024 -> 2048
- `impact`: 0.503063 -> 0.503063 (delta +0.000000)
- `duration`: 0.735394 -> 0.735394 (delta +0.000000)
- `peak_time`: 4.608556 -> 4.608556 (delta +0.000000)
- `mel_mean_db`: -32.158662 -> 3.213396 (delta +35.372059)
- `impact_freq_hz`: 1701.023364 -> 2526.139564 (delta +825.116199)

### `hop_length`: 4096 -> 2048
- `impact`: 0.503063 -> 0.503063 (delta +0.000000)
- `duration`: 0.735394 -> 0.735394 (delta +0.000000)
- `peak_time`: 4.608556 -> 4.608556 (delta +0.000000)
- `mel_mean_db`: -32.158662 -> -32.158662 (delta +0.000000)
- `impact_freq_hz`: 1701.023364 -> 1701.023364 (delta +0.000000)

## 25-split p-value table (winning params)

| split | impact | duration | peak_time | mel_mean_db | impact_freq_hz | best |
|---|---|---|---|---|---|---|
| A|B | 0.3204 | 0.0043 | 0.3054 | 0.5324 | 0.1377 | 0.0043 |
| A|C | 0.3441 | 0.8489 | 0.9637 | 0.4694 | 0.0815 | 0.0815 |
| A|D | 0.4185 | 0.6138 | 0.6629 | 0.8229 | 0.3695 | 0.3695 |
| B|C | 0.9534 | 0.0009 | 0.1957 | 0.1172 | 0.7201 | 0.0009 |
| B|D | 0.8647 | 0.0107 | 0.4652 | 0.3885 | 0.5155 | 0.0107 |
| C|D | 0.9653 | 0.4662 | 0.6670 | 0.4971 | 0.2946 | 0.2946 |
| A|BC | 0.2731 | 0.1694 | 0.6122 | 0.9101 | 0.0662 | 0.0662 |
| A|BD | 0.3034 | 0.0554 | 0.4041 | 0.8202 | 0.1740 | 0.0554 |
| A|CD | 0.3176 | 0.8833 | 0.8422 | 0.5767 | 0.1260 | 0.1260 |
| B|AC | 0.6482 | 0.0003 | 0.1747 | 0.1817 | 0.6134 | 0.0003 |
| B|AD | 0.6624 | 0.0017 | 0.3117 | 0.3819 | 0.2250 | 0.0017 |
| B|CD | 0.8987 | 0.0007 | 0.2362 | 0.1549 | 0.9016 | 0.0007 |
| C|AB | 0.5727 | 0.0288 | 0.4030 | 0.1663 | 0.2322 | 0.0288 |
| C|AD | 0.6182 | 0.5733 | 0.7675 | 0.4073 | 0.1038 | 0.1038 |
| C|BD | 0.9937 | 0.0173 | 0.3107 | 0.1868 | 0.4082 | 0.0173 |
| D|AB | 0.5836 | 0.1946 | 0.8326 | 0.5122 | 0.9299 | 0.1946 |
| D|AC | 0.6652 | 0.4663 | 0.6175 | 0.7407 | 0.8134 | 0.4663 |
| D|BC | 0.9060 | 0.3660 | 0.9031 | 0.9691 | 0.3218 | 0.3218 |
| A|BCD | 0.2732 | 0.2373 | 0.5958 | 0.8678 | 0.0962 | 0.0962 |
| B|ACD | 0.7978 | 0.0003 | 0.2071 | 0.1939 | 0.5339 | 0.0003 |
| C|ABD | 0.7077 | 0.0644 | 0.4387 | 0.2052 | 0.1950 | 0.0644 |
| D|ABC | 0.7057 | 0.6142 | 0.9505 | 0.9127 | 0.6672 | 0.6142 |
| AB|CD | 0.5015 | 0.0326 | 0.5108 | 0.2076 | 0.4129 | 0.0326 |
| AC|BD | 0.5846 | 0.0079 | 0.2539 | 0.3057 | 0.8691 | 0.0079 |
| AD|BC | 0.5707 | 0.1612 | 0.7050 | 0.9677 | 0.0836 | 0.0836 |
| BC|AD | 0.5707 | 0.1612 | 0.7050 | 0.9677 | 0.0836 | 0.0836 |
| BD|AC | 0.5846 | 0.0079 | 0.2539 | 0.3057 | 0.8691 | 0.0079 |
| CD|AB | 0.5015 | 0.0326 | 0.5108 | 0.2076 | 0.4129 | 0.0326 |

## Fisher ratio per feature

- `impact`: 0.3171
- `duration`: 4.4885
- `peak_time`: 1.1108
- `mel_mean_db`: 0.1942
- `impact_freq_hz`: 1.0159

## Mahalanobis centroid distance (per feature-pair, focus split)

- `AB|CD`: impact=0.004, duration=0.085, peak_time=0.040, mel_mean_db=0.019, impact_freq_hz=0.029
- `A|D`: impact=0.036, duration=0.031, peak_time=0.018, mel_mean_db=0.005, impact_freq_hz=0.037
- `B|C`: impact=0.032, duration=0.204, peak_time=0.082, mel_mean_db=0.036, impact_freq_hz=0.022
- `A|B`: impact=0.019, duration=0.161, peak_time=0.083, mel_mean_db=0.014, impact_freq_hz=0.068
- `C|D`: impact=0.049, duration=0.060, peak_time=0.018, mel_mean_db=0.037, impact_freq_hz=0.057

> Reproduce: `python3 dsp_inverse.py reproduce --out run1`