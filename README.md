<div align="center">

# 🧬 DNA Neural Context Compression 🧬

### Neural next-symbol prediction meets arithmetic coding for DNA sequences

<p>
  <img alt="Project status: research archive" src="https://img.shields.io/badge/status-research%20archive-6f42c1">
  <img alt="Reference round trip: passing" src="https://img.shields.io/badge/reference%20round--trip-passing-2ea44f">
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Upstream snapshot: pinned" src="https://img.shields.io/badge/upstream-fde1357-informational">
</p>

<p>
  <a href="#-research-question">Research question</a> •
  <a href="#-verification">Verification</a> •
  <a href="#-historical-results">Results</a> •
  <a href="#-reproducibility">Reproducibility</a> •
  <a href="#-third-party-attribution">Attribution</a>
</p>

</div>

This repository includes just the skeleton of the prior work.

## 🔬 Research question

Can small neural context models predict DNA symbols well enough to improve the
probability estimates used by arithmetic coding?

The experiments compare feed-forward, convolutional, and recurrent predictors
with fixed arithmetic coding, adaptive arithmetic coding, and prediction by
partial matching (PPM).

~~~mermaid
flowchart LR
    A["DNA context<br/>A C G T"] --> B{"Context model"}
    B --> C["CNN 1D"]
    B --> D["LSTM"]
    B --> E["FFNN"]
    C --> F["Next-symbol<br/>probabilities"]
    D --> F
    E --> F
    F --> G["Arithmetic coder"]
    G --> H["Compressed bitstream"]
~~~

Arithmetic coding can be lossless even when its probabilities come from a
neural network. The probabilities affect code length, while the arithmetic
coder records enough information to reconstruct every symbol—provided the
encoder and decoder use exactly the same deterministic probability sequence.

These models are neural context models, not hashing algorithms. Their purpose
is to estimate the next-symbol distribution used by the arithmetic coder.

## 📋 Overview

| Component | Status | What that means |
| :--- | :---: | :--- |
| Fixed arithmetic codec | ✅ Verified | Byte-for-byte round trip included |
| Adaptive arithmetic codec | ✅ Verified | Byte-for-byte round trip included |
| PPM codec | ✅ Verified | Byte-for-byte round trip included |
| CNN 1D / LSTM / FFNN models | 🧪 Experimental | Model definitions and training code preserved |
| Dataset and checkpoints | 📦 Not distributed | Local inputs and generated artifacts are excluded |
| Neural arithmetic integration | ⚠️ Incomplete | No verified end-to-end decompression test |
| Third-party provenance | 🔒 Pinned | 44 unmodified files from Project Nayuki commit <code>fde1357</code> |

## ✅ Verification

The verified reference-codec path has no third-party Python dependencies.
Immediately after cloning:

~~~bash
cd dna-neural-context-compression
python3 scripts/verify_reference_roundtrip.py
~~~

The script compresses temporary binary data with the fixed, adaptive, and PPM
Python implementations, decompresses it, and checks byte-for-byte equality.
A successful run ends with:

~~~text
Ran 1 test

OK
~~~

The same check is available through the Python standard library:

~~~bash
python3 -m unittest discover -s tests -v
~~~

## 🧠 Experimental models

The project represents DNA bases as categorical symbols and preserves three
small context models:

| Model | Context mechanism | Experimental role |
| :--- | :--- | :--- |
| CNN 1D | Local convolution over the sequence | Short-range motif/context modeling |
| LSTM | Recurrent hidden state | Sequential dependency modeling |
| FFNN | Per-position nonlinear projection | Lightweight probability baseline |

After installing the neural dependencies, the model-shape smoke test can be
run without a dataset or checkpoint:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test_neural_models.py
~~~

The historical neural encoder mixes a five-symbol DNA prediction model with a
byte-oriented arithmetic-coding interface. A matching, tested decompression
path was not completed. The neural code documents the experiment but is not
presented as a validated lossless codec.

## 📊 Historical results

The original experiment used a 1,591,049-byte DNA sample. The values below are
preserved for context and have not been independently reproduced in the cleaned
repository.

| Method | Compressed size | Ratio | Relative observation |
| --- | ---: | ---: | --- |
| Fixed arithmetic coding | 396,202 bytes | 24.90% | Reference baseline |
| Adaptive arithmetic coding | 395,623 bytes | 24.87% | Similar to fixed |
| **PPM** | **384,524 bytes** | **24.17%** | Best result in this experiment |
| CNN 1D | approximately 408 KB | approximately 26.22% | Slower; no improvement over PPM |
| LSTM | approximately 408 KB | approximately 26.22% | Slower; incomplete evaluation |
| FFNN | approximately 412 KB | approximately 26.47% | Lightweight neural baseline |

In these experiments, the neural predictors were slower and did not improve
compression over PPM. The result is useful as a negative experimental finding
and motivates better context models, deterministic inference, and stronger
round-trip validation.

<details>
<summary><strong>View the preserved result records</strong></summary>

- [Condensed experiment summary](results/summary.md)
- [Reference-codec measurements](results/reference_baselines.md)

</details>

## 🧪 Reproducibility

### Reference codecs

The fixed, adaptive, and PPM implementations are dependency-free and covered by
the fresh-clone round-trip check above.

### Neural environment

The neural experiments require Python 3.11, PyTorch, NumPy, pandas, and tqdm.
Choose either:

~~~bash
# Conda
conda env create -f environment.yml
conda activate dna-neural-context-compression
~~~

~~~bash
# Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The historical training scripts expect a user-supplied <code>DNACorpus/</code>
directory. Review paths and sampling settings before running them. GPU selection
is no longer tied to a particular machine, but exact reproduction of the
historical measurements is not guaranteed.

Model checkpoints are deliberately excluded. PyTorch checkpoint files should
only be loaded from trusted sources.

<details>
<summary><strong>Repository layout</strong></summary>

~~~text
.
├── experiments/
│   └── neural/             # Project-specific models and training code
├── results/                # Historical result summaries
├── scripts/                # Fresh-clone and model smoke checks
├── tests/                  # Automated reference-codec round trips
└── third_party/
    ├── provenance.json     # Machine-readable pinned provenance
    └── reference-arithmetic-coding/
                             # Unmodified 44-file upstream snapshot
~~~

</details>

## 🧭 Limitations and next steps

- Define one explicit DNA alphabet and input-normalization policy.
- Make neural inference deterministic across compression and decompression.
- Add a neural round-trip test covering empty, short, and long inputs.
- Record dataset provenance and produce reproducible benchmark splits.
- Compare compression ratio, encoding time, decoding time, and model size.

## 📚 References

- [Project Nayuki: Reference arithmetic coding, commit <code>fde1357</code>](https://github.com/nayuki/Reference-arithmetic-coding/commit/fde1357935494f395b4d17ca7e9e897c226ad208)
- [Efficient DNA sequence compression with neural networks](https://academic.oup.com/gigascience/article/9/11/giaa119/5974977)

## 📄 License

Project-specific source code is available under the MIT License in
[<code>LICENSE</code>](LICENSE). Third-party source remains under its original
license and copyright notice.

## 🔗 Third-party attribution

This project includes an unmodified 44-file snapshot of
[Project Nayuki's Reference arithmetic coding](https://github.com/nayuki/Reference-arithmetic-coding)
at commit
[<code>fde1357</code>](https://github.com/nayuki/Reference-arithmetic-coding/commit/fde1357935494f395b4d17ca7e9e897c226ad208).
That upstream code is copyright Project Nayuki and is distributed under the
MIT License. Its original copyright headers and license text are preserved in
<code>third_party/reference-arithmetic-coding/Readme.markdown</code>.

All files inside <code>third_party/reference-arithmetic-coding/</code> belong to
that pinned upstream snapshot. Project-specific additions are kept outside the
third-party directory. Machine-readable provenance, including the pinned Git
tree identifier, is recorded in <code>third_party/provenance.json</code>.

---

<div align="center">
  <sub>Preserved as an honest record of exploratory neural compression research.</sub>
</div>
