# Diagnosis (otomatis dari `diagnose.py`)

Utterance: `1089_134686_000001_000001` (speaker 1089, test-clean) - crop 4s di tengah.
Checkpoint: `checkpoints\best.pt`. 

## Tabel ukuran

| ukuran | real | roundtrip | meanmel | model |
|---|---|---|---|---|
| band_0-2k | 0.9606 | 0.9525 | 0.9602 | 0.9815 |
| band_2-4k | 0.03283 | 0.03779 | 0.03384 | 0.01024 |
| band_4-6k | 0.005358 | 0.008201 | 0.003253 | 0.004867 |
| band_6-8k | 0.0006575 | 0.0008289 | 0.001142 | 0.002552 |
| band_8-12k | 0.0005811 | 0.0006341 | 0.001522 | 0.0008055 |
| centroid_hz | 1726 | 1763 | 1801 | 1774 |
| hlr_4k | 0.006641 | 0.009759 | 0.005952 | 0.008292 |
| f0_mean_hz | 88.11 | n/a | n/a | 187.2 |
| f0_std_semitones | 1.547 | n/a | n/a | 2.507 |
| voiced_fraction | 0.3218 | 0 | 0 | 0.4249 |
| mel_delta_abs | 0.4899 | 0.4899 | 0.06871 | 0.5161 |
| mel_bin_std | 1.561 | 1.561 | 0.2255 | 1.734 |
| mel_range | 10.48 | 10.48 | 5.741 | 13.88 |
| mel_mean | -5.659 | -5.659 | -5.505 | -5.135 |

Audio pembanding ada di `diagnose_out_after/`: `*_real.wav`, `*_roundtrip.wav`, `meanmel.wav`, `*_model.wav`.

## Interpretasi otomatis

- Round-trip mempertahankan HLR 0.00976 vs asli 0.00664: pita tinggi relatif terjaga -> plafon bukan di vocoder
- Energi pita 8-11 kHz: asli nan, round-trip nan (nilai > 0 hanya mungkin bila mel_fmax > 8000)
- Dinamika antar-frame: asli 0.49 vs model 0.516 (rasio 1.05; << 1 berarti model terlalu halus / over-smoothing)
- Model vs mean-mel (jarak spektral rata-rata): 1.51 (kecil -> output mendekati 'suara rata-rata')
- Variasi F0: asli 1.55 semitone vs model 2.51 -> kontur intonasi masih wajar

## Cara membaca

| Ukuran | Arti | Arah yang diharapkan |
|---|---|---|
| `band_8-11k` | porsi energi 8-11 kHz | harus > 0 untuk konsonan tajam; 0 berarti `mel_fmax` memotong |
| `hlr_4k` | energi > 4 kHz / < 4 kHz | turun tajam = suara teredam |
| `centroid_hz` | "kecerahan" spektral | rendah = muffled |
| `f0_std_semitones` | variasi intonasi | rendah = monoton/zombie |
| `mel_delta_abs` | rata-rata |Δmel| antar frame | kecil = over-smoothing (blur) |
