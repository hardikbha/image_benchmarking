# DFToolBench-I: Benchmarking Tool-Augmented Agents for Image-Based Deepfake Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **Hardik Sharma, Prateek Shaily, Jayant Kumar, Praful Hambarde, Amit Shukla, Sachin Chaudhary**
>
> *Submitted to IEEE Transactions on Information Forensics and Security*

---

## Abstract

The rapid proliferation of sophisticated image manipulation techniques poses significant challenges to media forensics and content authentication. While large language models (LLMs) have shown remarkable reasoning capabilities, their effectiveness as tool-augmented agents for real-world deepfake detection remains largely uncharacterised.

We introduce **DFToolBench-I**, the first comprehensive benchmark designed to evaluate tool-augmented LLM agents on image-based deepfake detection tasks. The benchmark consists of **1,098 forensic queries** spanning four domains, a catalogue of **12 specialised forensic tools**, and a systematic evaluation of **35 LLMs** under both ReAct (agentic) and end-to-end (direct) settings.

Our evaluation reveals a substantial gap between current LLM capabilities and the demands of rigorous media forensics: even the strongest models score below 45 on step-level instruction accuracy, and under 35 on end-to-end answer accuracy. We further demonstrate a high Pearson correlation (r ≈ 0.919) between step-level and end-to-end metrics, validating our evaluation design. DFToolBench-I provides a reproducible framework—including datasets, tools, evaluation protocols, and scoring code—to drive progress in this critical application domain.

---

## Taxonomy of Forensic Domains

DFToolBench-I organises its 1,098 queries across four high-level forensic domains (Fig. 1 of the paper):

```
DFToolBench-I Taxonomy
├── Identity & Credential Forensics
│   ├── Face swap / deepfake detection
│   ├── Biometric anomaly localisation
│   └── ID-document authenticity
│
├── Document Forgery
│   ├── Text tampering & OCR forensics
│   ├── Seal / stamp forgery
│   └── Medical & financial records
│
├── Multi-evidence Aggregation
│   ├── Combined copy-move + splicing
│   ├── EXIF + content consistency
│   └── Fingerprint cross-checking
│
└── Cross-Domain Forensics
    ├── Scene change & temporal coherence
    ├── Object insertion / removal
    └── Blended multi-category attacks
```

Each domain exercises a distinct subset of the 12 tool catalogue, enabling fine-grained analysis of which forensic capabilities an agent is missing.

---

## Tool Catalogue

DFToolBench-I ships 12 forensic tools (Table III of the paper). Each tool wraps a state-of-the-art model behind a unified `BaseTool.apply()` interface and is served over HTTP during evaluation.

| Tool | Underlying Model | Reference |
|------|-----------------|-----------|
| `AnomalyDetectionTool` | PatchGuard (DINOv2 ViT-S/14 + Discriminator Transformer) | [PatchGuard](https://arxiv.org/abs/2312.02287) |
| `OCR` | EasyOCR (CRAFT detector + CRNN recogniser) | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| `TextForgeryLocalizerTool` | TruFor (CVPR 2023) | [TruFor](https://arxiv.org/abs/2305.10529) |
| `CopyMoveLocalizationTool` | CIML / ConvXL | [CIML/ConvXL](https://arxiv.org/abs/2401.07802) |
| `FaceDetectionTool` | EResFD (WACV 2024) | [EResFD](https://github.com/clovaai/EResFD) |
| `SceneChangeDetectionTool` | GeSCF (CVPR 2025) | [GeSCF](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Towards_Generalizable_Scene_Change_Detection_CVPR_2025_paper.html) |
| `FingerprintingTool` | Artificial GAN Fingerprints (ICCV 2021) | [GAN Fingerprints](https://arxiv.org/abs/2106.10021) |
| `Calculator` | Python `ast`-safe math evaluator | Built-in |
| `ObjectDetectionTool` | YOLOE / YOLOv8 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| `DeepfakeDetectionTool` | D3 (CLIP ViT-L/14, CVPR 2025) | [D3](https://arxiv.org/abs/2404.04584) |
| `SegmentationTool` | ASPC-Net (MIML) | [ASPC-Net](https://openaccess.thecvf.com/content/CVPR2024/html/Qu_Towards_Modern_Image_Manipulation_Localization_A_Large-Scale_Dataset_and_Novel_CVPR_2024_paper.html) |
| `DenoiseTool` | MaIR (CDNv25, CVPR 2025) | [MaIR](https://arxiv.org/abs/2407.10961) |

---

## Datasets

The benchmark draws images from the following publicly available datasets:

| Dataset | Description | Access |
|---------|-------------|--------|
| **CASIA v1 / v2** | Splicing & copy-move forgeries with pixel-level ground truth | [GitHub](https://github.com/namtpham/casia2groundtruth) |
| **IMD2020** | In-the-wild manipulated images with diverse forgery types | [Download](https://staff.utia.cas.cz/novozada/db/IMD2020.zip) |
| **SIDA** | Scene insertion & deletion attacks (CVPR 2025) | [arXiv](https://arxiv.org/abs/2501.07479) |
| **FantasyID / Fanta-ID** | Synthetic identity-document forgery (262 KYC-style ID cards, 13 templates) | [Zenodo](https://zenodo.org/records/17063366) / [arXiv](https://arxiv.org/abs/2507.20808) |
| **DefactoCopyMove** | ~19,000 copy-move forgeries over MS-COCO (EUSIPCO 2019) | [Website](https://defactodataset.github.io/) / [Kaggle](https://www.kaggle.com/datasets/defactodataset/defactocopymove) |
| **DoctorBills** | Medical bill document forgery (ICMM 2024) | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-53311-2_15) |
| **ChangeDetection** | Scene change detection benchmark (remote sensing) | [Paper](https://ieeexplore.ieee.org/document/7312067) / [Awesome-RSCD](https://github.com/wenhwu/awesome-remote-sensing-change-detection) |

### Benchmark Queries

The full set of **1,098 forensic queries** with structured annotations, ground-truth tool chains, and acceptance criteria is available for download:

| Resource | Description | Access |
|----------|-------------|--------|
| **DFToolBench-I Queries** | 1,098 human-authored, step-implicit queries with 12 executable tools | [GitHub Release](https://github.com/hardikbha/image_benchmarking/releases) |

> **Note:** Dataset download and placement instructions are provided in `dftoolbench/data/README.md` after cloning.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hardikbha/image_benchmarking.git
cd image_benchmarking
```

### 2. Create a base conda environment

```bash
conda create -n dftoolbench python=3.10 -y
conda activate dftoolbench
pip install -e ".[eval]"
```

### 3. Tool-specific environments

Several tools require isolated conda environments due to conflicting dependencies. Create them with the provided scripts:

```bash
# TruFor (TextForgeryLocalizerTool)
conda env create -f envs/trufor_env.yml

# CIML/ConvXL (CopyMoveLocalizationTool)
conda env create -f envs/ciml_env.yml

# GeSCF (SceneChangeDetectionTool)
conda env create -f envs/gescf_env.yml

# MaIR (DenoiseTool)
conda env create -f envs/mair_env.yml
```

Set the tool root so the server can locate CLI scripts and checkpoints:

```bash
export DFTOOLBENCH_TOOL_ROOT=/path/to/DFToolBench-I/tool_root
```

### 4. Download model checkpoints

```bash
python scripts/download_checkpoints.py --tools all
```

---

## Quick Start

### Running a single tool

```python
from dftoolbench.tools.ocr import OCRTool

tool = OCRTool(lang=["en"], device="cpu")
result = tool.apply("path/to/image.png")
print(result)
# (42,10,310,28) PATIENT NAME: John Doe
# (42,35,210,53) DATE: 2024-01-15
```

```python
from dftoolbench.tools.anomaly_detection import AnomalyDetectionTool
import json

tool = AnomalyDetectionTool(device="cuda:0")
result = json.loads(tool.apply("path/to/suspicious.jpg"))
print(result["anomaly_pct"])   # e.g. 7.3  (% anomalous pixels)
print(result["mask_path"])     # path to saved binary mask PNG
```

### Running the full evaluation pipeline

```bash
# 1. Start tool server (serves all 12 tools over HTTP)
bash scripts/start_tool_server.sh

# 2. Start bridge server (OpenCompass <-> tool server adapter)
bash scripts/start_bridge.sh

# 3. Run evaluation (edit MODEL and CONFIG to taste)
MODEL=gpt-4o \
CONFIG=configs/eval_configs/react_agent.py \
bash scripts/run_evaluation.sh

# All-in-one convenience wrapper
bash scripts/run_evaluation.sh --model gpt-4o --mode react
```

Results are written to `outputs/<run_id>/results.json`.

### LLM-as-a-Judge scoring

```python
from dftoolbench.evaluation import LLMJudge

judge = LLMJudge(model="gpt-4o", api_key="sk-...")

# Score a single agent trajectory
scores = judge.score_trajectory(
    trajectory_path="outputs/run_001/trajectories/q0001.json",
    rubric="inst_acc",          # one of: inst_acc, tool_acc, arg_acc, summ_acc
)
print(scores)
# {"inst_acc": 0.75, "tool_acc": 0.60, "arg_acc": 0.20, "summ_acc": 0.85}

# Batch score an entire run
from dftoolbench.evaluation import batch_score
batch_score(
    run_dir="outputs/run_001",
    output_csv="outputs/run_001/scores.csv",
)
```

---

## Evaluation Protocol

DFToolBench-I implements the two-tier evaluation framework described in Section V of the paper.

### Agent Modes

| Mode | Description |
|------|-------------|
| **ReAct** | LLM iteratively reasons, selects tools, observes results, and builds up an answer over multiple turns. |
| **E2E (direct)** | LLM receives the question and all tool outputs simultaneously and produces a single-turn answer. |

### Step-Level Metrics (ReAct mode)

These metrics are computed at the trajectory level using an LLM-as-a-Judge approach:

| Metric | Abbreviation | Description |
|--------|-------------|-------------|
| Instruction Accuracy | **InstAcc** | Does the agent's plan match the reference plan? |
| Tool Accuracy | **ToolAcc** | Does the agent select the correct tool at each step? |
| Argument Accuracy | **ArgAcc** | Does the agent supply correct arguments to each tool call? |
| Summary Accuracy | **SummAcc** | Does the agent correctly interpret and summarise tool output? |

### End-to-End Metrics

| Metric | Description |
|--------|-------------|
| **P&R** | Precision and Recall of forensic evidence mentioned in the final answer |
| **ML&A** | Multi-label answer accuracy across all forgery attributes |
| **QR&A** | Question–answer relevance score |
| **AnsAcc** | Overall binary answer accuracy (correct/incorrect final verdict) |

### Metric Correlation

A Pearson correlation of **r ≈ 0.919** between step-level and end-to-end metrics validates that strong agentic reasoning translates to strong final answers, and that step-level evaluation is a reliable proxy for end-to-end performance.

---

## Results

**Table IV** — Selected results from the 35-model evaluation (higher is better).

| Model | InstAcc | ToolAcc | ArgAcc | SummAcc | P&R | ML&A | QR&A | AnsAcc |
|-------|--------:|--------:|-------:|--------:|----:|-----:|-----:|-------:|
| GPT-4 | 29.5 | 17.5 | 2.0 | 41.2 | 24.8 | 22.6 | 21.2 | 4.5 |
| GPT-4.1 | 35.4 | 22.1 | 2.8 | 41.2 | 29.7 | 28.3 | 25.6 | 5.8 |
| GPT-4o | 37.0 | 22.8 | 2.8 | 41.2 | 11.1 | 28.6 | 26.6 | 5.3 |
| GPT-5 | **41.0** | **26.5** | **3.3** | **56.5** | **34.4** | **32.8** | **29.5** | **7.2** |
| Claude-4.1-Opus | 28.8 | 16.2 | 1.9 | 35.8 | 23.6 | 22.6 | 20.2 | 5.7 |
| Claude-4.0-Sonnet | 33.2 | 20.0 | 2.3 | 51.8 | 27.9 | 26.6 | 20.5 | 6.2 |

> Full results for all 35 models are available in `outputs/full_results.csv` and in the paper (Table IV).

**Key findings:**
- Even the top-performing model (GPT-5) achieves only 41.0 InstAcc and 7.2 AnsAcc, revealing a wide performance gap.
- ArgAcc is consistently the weakest metric (< 4 for all models), highlighting tool-calling precision as the primary bottleneck.
- The high P/r correlation (0.919) confirms that step-level metrics are reliable proxies for end-to-end performance.

---

## Project Structure

```
DFToolBench-I/
├── configs/
│   └── eval_configs/          # OpenCompass evaluation configs
│       ├── react_agent.py
│       └── e2e_direct.py
│
├── dftoolbench/               # Core Python package
│   ├── __init__.py
│   ├── data/                  # Dataset loaders & query builders
│   │   └── __init__.py
│   ├── evaluation/            # Metrics, judge, batch scoring
│   │   └── __init__.py
│   └── tools/                 # All 12 tool implementations
│       ├── base.py
│       ├── ocr.py
│       ├── anomaly_detection.py
│       ├── text_forgery_localizer.py
│       ├── copy_move_localization.py
│       ├── face_detection.py
│       ├── scene_change_detection.py
│       ├── fingerprinting.py
│       ├── calculator.py
│       ├── object_detection.py
│       ├── deepfake_detection.py
│       ├── segmentation.py
│       └── denoise.py
│
├── envs/                      # Per-tool conda environment files
│   ├── trufor_env.yml
│   ├── ciml_env.yml
│   ├── gescf_env.yml
│   └── mair_env.yml
│
├── examples/
│   └── sample_query.json      # Two annotated example queries
│
├── scripts/
│   ├── run_evaluation.sh      # Main evaluation entry point
│   ├── start_tool_server.sh   # Launch HTTP tool server
│   ├── start_bridge.sh        # Launch OpenCompass bridge
│   ├── download_checkpoints.py
│   └── smoke_test.py          # Quick sanity check
│
├── tool_root/                 # CLI scripts & checkpoint dirs (gitignored)
│   ├── patchguard_cli.py
│   ├── trufor_cli.py
│   └── checkpoints/
│
├── outputs/                   # Evaluation results (gitignored)
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Citation

The paper is currently under review. Citation details will be updated upon publication.

---

## License

This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

Individual tool models and datasets retain their own licences; please consult each upstream repository before commercial use.

---

## Acknowledgments

DFToolBench-I builds on top of several excellent open-source frameworks:

- **[GTA Benchmark](https://github.com/open-compass/GTA)** — General Tool-Augmented agent benchmark framework that inspired our evaluation design.
- **[OpenCompass](https://github.com/open-compass/opencompass)** — LLM evaluation framework used to orchestrate model inference and scoring.
- **[AgentLego](https://github.com/open-mmlab/agentlego)** — Tool-augmented agent library whose tool-server architecture we adapted for forensic tools.

We also thank the authors of all upstream models (TruFor, CIML/ConvXL, PatchGuard, EResFD, GeSCF, D3, MaIR, EasyOCR, YOLOE) for releasing their code and weights publicly.
