# Fourier Inception Gated Super Resolution

The main idea of the model is to integrate the [FourierUnit](https://github.com/deng-ai-lab/SFHformer/blob/1f7994112b9ced9153edc7187e320e0383a9dfd3/models/SFHformer.py#L143) into the [GatedCNN](https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119) pipeline in order to strengthen the model’s global perception with minimal computational overhead.

The FourierUnit adds feature processing in the frequency domain, expanding the effective receptive field, while the GatedCNN provides efficient local modeling and control of information flow through a gating mechanism. Their combination allows merging global context and computational efficiency within a compact SISR architecture.

---

## Model structure:

### FIDSR

<img src="figs/FIDSR.png" width="600"/>

### GDB FU

<img src="figs/gdb_and_FU.png" width="600"/>

---

### Main blocks and their changes relative to the originals:

* [GatedCNN](https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119) — borrowed from the [MambaOut](https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119) repository with the following changes:

  * `Linear` replaced with `Conv` to avoid unnecessary `permute` operations;
  * one of the linear layers replaced with a `Conv 3×3`, which improves quality without a significant increase in computational cost;
  * `LayerNorm` replaced with `RMSNorm` for speed and greater stability;
  * `DConv` replaced with `InceptionConv`.

* [InceptionConv](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L627) — a modified version of the block from [InceptionNeXt](https://github.com/sail-sg/inceptionnext/blob/main/models/inceptionnext.py#L19):

  * `DConv` replaced with standard convolutions;
  * kernel sizes increased following the findings of [PLKSR](https://github.com/dslisleedh/PLKSR);
  * the shortcut replaced with `FourierUnit`, which improves convergence because a residual connection is already present inside `GatedCNN`.

* [FourierUnit](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L585) — a modified version of the block from [SFHformer](https://github.com/deng-ai-lab/SFHformer/blob/1f7994112b9ced9153edc7187e320e0383a9dfd3/models/SFHformer.py#L143):

  * `BatchNorm` replaced with `RMSNorm`, which works better with the small batch sizes typical for SISR;
  * structural changes made for correct export to ONNX;
  * post-normalization added, since without it training instability and `NaN` values were observed in the context of `GatedCNN`.

---

## Metrics:

| Model name | Params | Latency | Train dataset | urban100       | set14          | BHI100 | PSISRD_val125 | 
|------------| ------ |---------|---------------| -------------- | -------------- | ------ | ------------- | 
| ESRGAN     | ~16.5m |         | DF2K          | 27.03 / 0.8153 | 28.99 / 0.7917 |        |               |
| FIGSR      | ~4.4m  |         | BHI           | 27.11 / 0.8146 |                |        |               |

---

## Training

To train, choose one of the frameworks and place the model file in the `archs` folder:

* **[NeoSR](https://github.com/neosr-project/neosr)** — `fidsr_arcg.py` → `neosr/archs/fidsr_arcg.py`. [Config](configs/neosr.toml)

  * Uncomment lines [14–17](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L14-L17), [687](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L687) and [698](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L698).
  * Comment out line [696](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L696).

* **[traiNNer-redux](https://github.com/the-database/traiNNer-redux)** — `fidsr_arcg.py` → `traiNNer/archs/fidsr_arcg.py`. [Config](configs/trainner-redux.yml)

  * Uncomment lines [11](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L11) and [687](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L687).

* **[BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/archs)** — `fidsr_arcg.py` → `basicsr/archs/fidsr_arcg.py`. [Config](configs/basicsr.yml)

  * Uncomment lines [19](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L19) and [687](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L687).

---

## Inference:

TODO
