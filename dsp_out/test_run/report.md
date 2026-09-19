# DSP Inverse Optimizer - Report

- **Mode**: `generic` (focus split `AB|CD|AB`)
- **Best DSP params**: `sr=384000`, `n_fft=1024`, `hop_length=512`, `top_db=19`, `window=0.45s`
- **Objective** (trial 7/8): `2.66318`
- **Best overall split**: `B|ACD` p=0.0021 @ `duration`
- **Silhouette** (z-scored core features): -0.0392
- **Files used**: 321  (A=67 B=79 C=96 D=79)

## Feature importance (one-at-a-time param sensitivity)

| param | from -> to | objective delta | most-moved features |
|---|---|---|---|
| sr | 384000 -> 192000 | -0.8164 | impact_freq_hz, mel_mean_db, impact |
| n_fft | 1024 -> 512 | +0.0023 | impact_freq_hz, mel_mean_db, impact |
| n_fft | 1024 -> 2048 | +0.0021 | impact_freq_hz, mel_mean_db, impact |
| hop_length | 512 -> 256 | +0.0000 | impact, duration, peak_time |
| window_sec | 0.45 -> 0.4 | +0.0000 | impact, duration, peak_time |
| hop_length | 512 -> 1024 | +0.0000 | impact, duration, peak_time |
| window_sec | 0.45 -> 0.5 | +0.0000 | impact, duration, peak_time |
| top_db | 19 -> 18 | +0.0000 | impact, duration, peak_time |

## Mathematical relationship (parameter -> feature -> separation)

### `sr`: 384000 -> 192000
- `impact`: 0.502606 -> 0.479642 (delta -0.022965)
- `duration`: 0.622272 -> 0.634185 (delta +0.011913)
- `peak_time`: 4.962611 -> 4.959 (delta -0.003611)
- `mel_mean_db`: -31.433404 -> -1.908782 (delta +29.524622)
- `impact_freq_hz`: 1842.980685 -> 1503.424299 (delta -339.556386)

### `n_fft`: 1024 -> 512
- `impact`: 0.502606 -> 0.502606 (delta +0.000000)
- `duration`: 0.622272 -> 0.622272 (delta +0.000000)
- `peak_time`: 4.962611 -> 4.962611 (delta +0.000000)
- `mel_mean_db`: -31.433404 -> -75.689117 (delta -44.255713)
- `impact_freq_hz`: 1842.980685 -> 1893.814019 (delta +50.833333)

### `n_fft`: 1024 -> 2048
- `impact`: 0.502606 -> 0.502606 (delta +0.000000)
- `duration`: 0.622272 -> 0.622272 (delta +0.000000)
- `peak_time`: 4.962611 -> 4.962611 (delta +0.000000)
- `mel_mean_db`: -31.433404 -> 3.922962 (delta +35.356366)
- `impact_freq_hz`: 1842.980685 -> 2580.482866 (delta +737.502181)

### `hop_length`: 512 -> 256
- `impact`: 0.502606 -> 0.502606 (delta +0.000000)
- `duration`: 0.622272 -> 0.622272 (delta +0.000000)
- `peak_time`: 4.962611 -> 4.962611 (delta +0.000000)
- `mel_mean_db`: -31.433404 -> -31.433404 (delta +0.000000)
- `impact_freq_hz`: 1842.980685 -> 1842.980685 (delta +0.000000)

## 25-split p-value table (winning params)

| split | impact | duration | peak_time | mel_mean_db | impact_freq_hz | best |
|---|---|---|---|---|---|---|
| A|B | 1.0000 | 1.0000 | 0.2767 | 0.3262 | 0.2427 | 0.2427 |
| A|C | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1228 | 0.1228 |
| A|D | 1.0000 | 1.0000 | 0.8845 | 1.0000 | 0.0230 | 0.0230 |
| B|C | 1.0000 | 0.0137 | 1.0000 | 1.0000 | 0.8326 | 0.0137 |
| B|D | 1.0000 | 0.0246 | 1.0000 | 1.0000 | 0.4319 | 0.0246 |
| C|D | 1.0000 | 1.0000 | 0.7359 | 0.3306 | 0.4234 | 0.3306 |
| A|BC | 1.0000 | 1.0000 | 0.6398 | 1.0000 | 0.1210 | 0.1210 |
| A|BD | 1.0000 | 1.0000 | 0.4812 | 0.6088 | 0.0495 | 0.0495 |
| A|CD | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0333 | 0.0333 |
| B|AC | 0.6752 | 0.0023 | 1.0000 | 1.0000 | 1.0000 | 0.0023 |
| B|AD | 0.7081 | 0.0031 | 1.0000 | 1.0000 | 1.0000 | 0.0031 |
| B|CD | 1.0000 | 0.0067 | 1.0000 | 1.0000 | 0.5837 | 0.0067 |
| C|AB | 0.5264 | 1.0000 | 0.3370 | 0.0732 | 1.0000 | 0.0732 |
| C|AD | 0.6329 | 0.9059 | 0.7618 | 0.2621 | 1.0000 | 0.2621 |
| C|BD | 0.9768 | 1.0000 | 0.3107 | 0.0763 | 0.7299 | 0.0763 |
| D|AB | 0.4796 | 1.0000 | 0.5850 | 0.4703 | 1.0000 | 0.4703 |
| D|AC | 0.5929 | 0.5767 | 1.0000 | 1.0000 | 1.0000 | 0.5767 |
| D|BC | 0.8219 | 1.0000 | 0.7412 | 1.0000 | 1.0000 | 0.7412 |
| A|BCD | 1.0000 | 1.0000 | 0.6944 | 1.0000 | 0.0482 | 0.0482 |
| B|ACD | 0.8620 | 0.0021 | 1.0000 | 1.0000 | 1.0000 | 0.0021 |
| C|ABD | 0.6892 | 1.0000 | 0.4120 | 0.0928 | 1.0000 | 0.0928 |
| D|ABC | 0.6128 | 1.0000 | 0.8434 | 0.9872 | 1.0000 | 0.6128 |
| AB|CD | 1.0000 | 0.1760 | 1.0000 | 1.0000 | 0.1070 | 0.1070 |
| AC|BD | 1.0000 | 1.0000 | 0.2841 | 0.1372 | 0.1966 | 0.1372 |
| AD|BC | 1.0000 | 1.0000 | 0.6226 | 1.0000 | 0.7521 | 0.6226 |
| BC|AD | 0.6040 | 0.0818 | 1.0000 | 0.9697 | 1.0000 | 0.0818 |
| BD|AC | 0.5576 | 0.0269 | 1.0000 | 1.0000 | 1.0000 | 0.0269 |
| CD|AB | 0.4202 | 1.0000 | 0.3546 | 0.1188 | 1.0000 | 0.1188 |

## Fisher ratio per feature

- `impact`: 0.3651
- `duration`: 3.4545
- `peak_time`: 0.9002
- `mel_mean_db`: 0.1193
- `impact_freq_hz`: 1.4221

## Mahalanobis centroid distance (per feature-pair, focus split)

- `AB|CD`: impact=0.005, duration=0.066, peak_time=0.040, mel_mean_db=0.016, impact_freq_hz=0.022
- `A|D`: impact=0.039, duration=0.037, peak_time=0.009, mel_mean_db=0.004, impact_freq_hz=0.112
- `B|C`: impact=0.033, duration=0.162, peak_time=0.078, mel_mean_db=0.026, impact_freq_hz=0.036
- `A|B`: impact=0.020, duration=0.158, peak_time=0.072, mel_mean_db=0.016, impact_freq_hz=0.101
- `C|D`: impact=0.053, duration=0.037, peak_time=0.016, mel_mean_db=0.022, impact_freq_hz=0.027

> Reproduce: `python3 dsp_inverse.py reproduce --out test_run`