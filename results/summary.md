# Historical result summary

These measurements are preserved from the original experiments. They have not
been independently reproduced in the cleaned repository and should be treated
as historical observations rather than a current benchmark.

| Dataset | Original | Fixed arithmetic | Adaptive arithmetic | PPM |
| --- | ---: | ---: | ---: | ---: |
| AeCa | 1,591,049 bytes | 396,202 bytes | 395,623 bytes | 384,524 bytes |

The original neural experiments reported approximately 408 KB for the CNN and
LSTM models and approximately 412 KB for the FFNN models on the same sample.
Training and inference settings were exploratory, and no neural decompression
round-trip result was recorded.
