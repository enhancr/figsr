# Fourier Inception Gated Super Resolution

The main idea of the model is to integrate the [FourierUnit](https://github.com/deng-ai-lab/SFHformer/blob/1f7994112b9ced9153edc7187e320e0383a9dfd3/models/SFHformer.py#L143) into the [GatedCNN](https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119) pipeline in order to strengthen the model’s global perception with minimal computational overhead.

The FourierUnit adds feature processing in the frequency domain, expanding the effective receptive field, while the GatedCNN provides efficient local modeling and control of information flow through a gating mechanism. Their combination allows merging global context and computational efficiency within a compact SISR architecture.

---
# TODO:
+ [ ] Fix trt inference
---
## Showcase:
[show pics](https://slow.pics/s/fPvcS3P0?image-fit=contain) 

[gdrive](https://drive.google.com/drive/u/1/folders/1ofJo5CCgrOtLdVm9psmlJv15Z3aP4Aiz)

---
## Model structure:

### figsr

<img src="figs/figsr.png" width="600"/>

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
* Metrics were computed using [PyIQA](https://github.com/chaofengc/IQA-PyTorch/tree/main), except for those starting with “bs”, which were calculated using BasicSR.
### [Esrgan DF2K](https://drive.google.com/file/d/1mSJ6Z40weL-dnPvi390xDd3uZBCFMeqr/view?usp=sharing):
| Dataset       | SSIM-Y | PSNR-Y | TOPIQ  | bs_ssim_y | bs_psnr_y |
| ------------- | ------ | ------ | ------ | --------- | --------- |
| BHI100        | 0.7150 | 22.84  | 0.5694 | 0.7279    | 24.1636   |
| psisrd_val125 | 0.7881 | 27.01  | 0.6043 | 0.8034    | 28.3273   |
| set14         | 0.7730 | 27.67  | 0.6905 | 0.7915    | 28.9969   |
| urban100      | 0.8025 | 25.71  | 0.6701 | 0.8152    | 27.0282   |
### [FIGSR BHI](https://github.com/enhancr/figsr/releases/tag/v1.0.0):
| Dataset       | SSIM-Y | PSNR-Y | TOPIQ  | bs_ssim_y | bs_psnr_y |
| ------------- | ------ | ------ | ------ | --------- | --------- |
| BHI100        | 0.7196 | 22.83  | 0.5723 | 0.7327    | 24.1549   |
| psisrd_val125 | 0.7911 | 26.97  | 0.6095 | 0.8065    | 28.2946   |
| set14         | 0.7769 | 27.70  | 0.7036 | 0.7952    | 29.0221   |
| urban100      | 0.8056 | 25.80  | 0.6725 | 0.8185    | 27.1170   |

---

## Performance 3060 12gb:
| Model  | input_size | params ↓ | avg_inference ↓ | fps ↑              | memory_use ↓ |
|--------| ---------- | -------- |-----------------| ------------------ | ------------ |
| ESRGAN | 1024x1024  | ~16.6m   | ~2.8s           | 0.3483220866736526 | 8.29GB       |
| FIGSR  | 1024x1024  | ~4.4m    | ~1.64s          | 0.6081749253740837 | 2.26GB       |

## Training

To train, choose one of the frameworks and place the model file in the `archs` folder:

* **[NeoSR](https://github.com/neosr-project/neosr)** — `figsr_arch.py` → `neosr/archs/figsr_arch.py`. [Config](configs/neosr.toml)

  * Uncomment lines [14–17](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L14-L17), [694](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L694) and [705](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L705).
  * Comment out line [703](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L703).

* **[traiNNer-redux](https://github.com/the-database/traiNNer-redux)** — `figsr_arch.py` → `traiNNer/archs/figsr_arch.py`. [Config](configs/trainner-redux.yml)

  * Uncomment lines [11](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L11) and [694](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L694).

* **[BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/archs)** — `figsr_arch.py` → `basicsr/archs/figsr_arch.py`. [Config](configs/basicsr.yml)

  * Uncomment lines [19](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L19) and [694](https://github.com/enhancr/figsr/blob/main/figsr_arch.py#L694).

---

## Inference:
### Resselt install
```shell
uv venv  --python=3.12
source .venv/bin/activate
uv pip install "resselt==1.3.1" "pepeline==1.2.3"
```
### main.py
```shell
 python main.py --input_dir urban/x4 --output_dir urban/x4_scale --weights  4x_FIGSR.safetensors 
```
---
## Contacts:
[discord](https://discord.gg/xwZfWWMwBq)