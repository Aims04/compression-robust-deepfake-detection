Videos: 391 (real 77, fake 314); 5-fold grouped CV x 5 repeats; embedder clip-vit-base-patch16; detector yunet.

| Training | Test level | AUC | Balanced acc | ECE (calibrated) | ECE (raw) |
|---|---|---|---|---|---|
| clean-trained | orig | 0.864 ± 0.042 | 0.731 ± 0.054 | 0.103 | 0.097 |
| clean-trained | crf28 | 0.841 ± 0.051 | 0.692 ± 0.055 | 0.107 | 0.108 |
| clean-trained | crf35 | 0.800 ± 0.052 | 0.603 ± 0.050 | 0.114 | 0.130 |
| clean-trained | social | 0.716 ± 0.069 | 0.538 ± 0.040 | 0.136 | 0.159 |
| clean-trained | crf42 | 0.666 ± 0.087 | 0.549 ± 0.042 | 0.144 | 0.174 |
| compression-aug-trained | orig | 0.865 ± 0.042 | 0.756 ± 0.062 | 0.094 | 0.112 |
| compression-aug-trained | crf28 | 0.864 ± 0.043 | 0.741 ± 0.061 | 0.101 | 0.117 |
| compression-aug-trained | crf35 | 0.848 ± 0.051 | 0.730 ± 0.070 | 0.098 | 0.119 |
| compression-aug-trained | social | 0.790 ± 0.049 | 0.682 ± 0.071 | 0.114 | 0.140 |
| compression-aug-trained | crf42 | 0.763 ± 0.062 | 0.647 ± 0.066 | 0.113 | 0.137 |
