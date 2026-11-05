# Diagnosis (otomatis dari `diagnose.py`)

Utterance: `1089_134686_000001_000001` (speaker 1089, test-clean) — crop 4s di tengah.
Checkpoint: `checkpoints\best.pt`. 

## Tabel ukuran

| ukuran | real | roundtrip | meanmel | model |
|---|---|---|---|---|
| band_0-2k | 0.9606 | 0.9525 | 0.9509 | 0.969 |
| band_2-4k | 0.03283 | 0.03779 | 0.03825 | 0.02212 |
| band_4-6k | 0.005358 | 0.008201 | 0.004955 | 0.004859 |
| band_6-8k | 0.0006575 | 0.0008289 | 0.002067 | 0.002608 |
| band_8-12k | 0.0005811 | 0.0006341 | 0.003857 | 0.001378 |
| centroid_hz | 1726 | 1763 | 2026 | 1866 |
| hlr_4k | 0.006641 | 0.009759 | 0.011 | 0.008924 |
| f0_mean_hz | 88.11 | n/a | n/a | 214.3 |
| f0_std_semitones | 1.547 | n/a | n/a | 1.018 |
| voiced_fraction | 0.3218 | 0 | 0 | 0.09538 |
| mel_delta_abs | 0.4899 | 0.4899 | 0.04201 | 0.4514 |
| mel_bin_std | 1.561 | 1.561 | 0.1328 | 1.782 |
| mel_range | 10.48 | 10.48 | 5.178 | 13.46 |
| mel_mean | -5.659 | -5.659 | -5.483 | -5.183 |

Audio pembanding ada di `diagnose_out/`: `*_real.wav`, `*_roundtrip.wav`, `meanmel.wav`, `*_model.wav`.

## Interpretasi otomatis

- Round-trip mempertahankan HLR 0.00976 vs asli 0.00664: pita tinggi relatif terjaga → plafon bukan di vocoder
- Energi pita 8-11 kHz: asli nan, round-trip nan (nilai > 0 hanya mungkin bila mel_fmax > 8000)
- Dinamika antar-frame: asli 0.49 vs model 0.451 (rasio 0.92; << 1 berarti model terlalu halus / over-smoothing)
- Model vs mean-mel (jarak spektral rata-rata): 1.65 (kecil → output mendekati 'suara rata-rata')
- Variasi F0: asli 1.55 semitone vs model 1.02 → kontur intonasi masih wajar

## Cara membaca

| Ukuran | Arti | Arah yang diharapkan |
|---|---|---|
| `band_8-11k` | porsi energi 8-11 kHz | harus > 0 untuk konsonan tajam; 0 berarti `mel_fmax` memotong |
| `hlr_4k` | energi > 4 kHz / < 4 kHz | turun tajam = suara teredam |
| `centroid_hz` | "kecerahan" spektral | rendah = muffled |
| `f0_std_semitones` | variasi intonasi | rendah = monoton/zombie |
| `mel_delta_abs` | rata-rata |Δmel| antar frame | kecil = over-smoothing (blur) |
