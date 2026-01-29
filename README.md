# ColPali: Efficient Document Retrieval with Vision Language Models 👀

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![arXiv](https://img.shields.io/badge/arXiv-ModernVBERT-2510.01149-b31b1b.svg?style=for-the-badge)](https://arxiv.org/html/2510.01149v1)
[![GitHub](https://img.shields.io/badge/ViDoRe_Benchmark-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/vidore)
[![GitHub](https://img.shields.io/badge/Cookbooks-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tonywu71/colpali-cookbooks)

[![Test](https://github.com/illuin-tech/colpali/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/illuin-tech/colpali/actions/workflows/test.yml)
[![Version](https://img.shields.io/pypi/v/colpali-engine?color=%2334D058&label=pypi%20package)](https://pypi.org/project/colpali-engine/)
[![Downloads](https://static.pepy.tech/badge/colpali-engine)](https://pepy.tech/project/colpali-engine)

---

> **Note**: This is a modified version of the [ColPali repository](https://github.com/illuin-tech/colpali) focused on training and evaluating **ColModernVBERT** models. The original repository supports multiple vision retrieval models, while this fork is specialized for the ModernVBERT architecture.

[[Model card]](https://huggingface.co/ModernVBERT/colmodernvbert)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
[[Demo]](https://huggingface.co/spaces/manu/ColPali-demo)
[[Blog Post]](https://huggingface.co/blog/manu/colpali)

## Associated Papers

This repository contains the code used for training vision retrievers, with a focus on **ColModernVBERT** from the [*ModernVBERT: Towards Smaller Visual Document Retrievers*](https://arxiv.org/html/2510.01149v1) paper. It is based on the [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://arxiv.org/abs/2407.01449) framework.

## Introduction

**ColModernVBERT** is a compact 250M-parameter vision-language encoder that outperforms models up to 10 times larger when finetuned on document retrieval tasks. It leverages the ColBERT late-interaction architecture with ModernVBERT, a model that combines a pretrained language encoder (ModernBERT) with a vision encoder (SigLIP-2) through Masked Language Modeling (MLM) alignment.

The model is trained in three phases:
1. **Phase 1**: Text-only ModernBERT base model (`answerdotai/ModernBERT-base`)
2. **Phase 2**: Vision-language alignment (`ModernVBERT/modernvbert`) - SigLIP-2 connected to ModernBERT using 2.5B image-text pairs
3. **Phase 3**: Contrastive fine-tuning for retrieval (`ModernVBERT/colmodernvbert`) - The final retrieval model

Using ColModernVBERT removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

![ColPali Architecture](assets/colpali_architecture.webp)

## Supported Models

| Model | Score on [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) 🏆 | License | Comments | Currently supported |
|-------|-------------------------------------------------------------------------------|---------|----------|---------------------|
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | 81.2 (v1) / 56.0 (v2) | Apache 2.0 | • 250M parameters<br />• Based on ModernBERT + SigLIP-2<br />• Multi-vector late-interaction retrieval | ✅ |

## Setup

We used Python 3.11.6 and PyTorch 2.4 to train and test our models, but the codebase is compatible with Python >=3.9 and recent PyTorch versions. To install the package, run:

```bash
pip install colpali-engine # from PyPi
pip install git+https://github.com/illuin-tech/colpali # from source
```

Mac users using MPS with the ColQwen models have reported errors with torch 2.6.0. These errors are fixed by downgrading to torch 2.5.1.

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.

## Usage

### Quick start

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

model_name = "ModernVBERT/colmodernvbert"

model = ColModernVBert.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColModernVBertProcessor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

### Benchmarking

To benchmark ColModernVBERT on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

### Training

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script. You can do this using the following command:

```bash
pip install "colpali-engine[train]"
```

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

#### Training ColModernVBERT

This repository is configured to train ColModernVBERT starting from the Phase 3 checkpoint (`ModernVBERT/colmodernvbert`). Two training configuration files are available:

1. **`train_colmodernvbert_16gb.yaml`**: Optimized for single GPU with 16GB VRAM, uses custom training hyperparameters
2. **`train_colmodernvbert_standard.yaml`**: Follows the original ColPali standard format, uses default ColPali training arguments and infrastructure

**Important**: Before training, set the environment variable to load datasets from HuggingFace:

```bash
# Linux/Mac
export USE_LOCAL_DATASET=0

# Windows
set USE_LOCAL_DATASET=0
```

##### Available Configurations

| Config File | Best For | Key Features |
|------------|----------|--------------|
| `train_colmodernvbert_16gb.yaml` | Single GPU with limited VRAM | • Custom training hyperparameters (3 epochs, batch size 1, gradient accumulation 8)<br />• Direct model instantiation<br />• Inline training arguments |
| `train_colmodernvbert_standard.yaml` | Compatibility with original ColPali infrastructure | • Follows original ColPali standard format<br />• Uses `ColModelTrainingConfig` wrapper<br />• Uses `AllPurposeWrapper` for model/processor<br />• Uses `!import` directives for training args and eval datasets<br />• Uses default ColPali training arguments (1 epoch, batch size 4) |

Both configs support Phase 1 training (filtered by `source="tatdqa"`) and can be modified for Phase 2 (multi-dataset training).

##### Training Commands

**Using the 16GB optimized config**:

```bash
python scripts/train/train_colbert.py scripts/configs/train_colmodernvbert_16gb.yaml
```

**Using the standard ColPali config**:

```bash
python scripts/train/train_colbert.py scripts/configs/train_colmodernvbert_standard.yaml
```

Or with accelerate for multi-GPU training:

```bash
# 16GB config
accelerate launch scripts/train/train_colbert.py scripts/configs/train_colmodernvbert_16gb.yaml

# Standard config
accelerate launch scripts/train/train_colbert.py scripts/configs/train_colmodernvbert_standard.yaml
```

##### Configuration Details

**16GB Optimized Config** (`train_colmodernvbert_16gb.yaml`):
- **Model**: Starts from `ModernVBERT/colmodernvbert` (Phase 3 checkpoint)
- **Training dataset**: `vidore/colpali_train_set` (loaded from HuggingFace, filtered by `source="tatdqa"` for Phase 1)
- **Evaluation datasets**: Single dataset (`syntheticDocQA_energy_test`) or all 10 ViDoRe test datasets (configured in `scripts/configs/data/test_data.yaml`)
- **Training epochs**: 3
- **Effective batch size**: 8 (per_device_train_batch_size: 1 × gradient_accumulation_steps: 8)
- **Learning rate**: 5e-5
- **Mixed precision**: bfloat16
- **Gradient checkpointing**: Enabled
- **Output directory**: `./output/colmodernvbert/`

**Standard ColPali Config** (`train_colmodernvbert_standard.yaml`):
- **Model**: Starts from `ModernVBERT/colmodernvbert` (Phase 3 checkpoint)
- **Training dataset**: `vidore/colpali_train_set` (loaded from HuggingFace, filtered by `source="tatdqa"` for Phase 1)
- **Evaluation datasets**: All 10 ViDoRe test datasets (configured in `scripts/configs/data/test_data.yaml`)
- **Training epochs**: 1 (from default ColPali training args)
- **Effective batch size**: 4 (per_device_train_batch_size: 4)
- **Learning rate**: 5e-5
- **Mixed precision**: bfloat16 (via `!ext torch.bfloat16`)
- **Output directory**: `../../../output/colmodernvbert_standard/` (relative to config file)
- **Config structure**: Uses `ColModelTrainingConfig` wrapper, `AllPurposeWrapper`, and `!import`/`!ext` directives

The training script will:
1. Load the pretrained ColModernVBERT model
2. Train on the colpali_train_set (filtered by source for Phase 1)
3. Evaluate on test datasets during training
4. Save checkpoints to the configured output directory

**After training**:
- The trained model will be saved in the configured output directory
- To evaluate on test splits, use the `vidore-benchmark` package
- To benchmark on the benchmark split, submit to the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)

## Contributing

This is a modified version of the ColPali repository. For contributions to the main ColPali project, please refer to the [original repository](https://github.com/illuin-tech/colpali).

## Community Projects

Several community projects and ressources have been developed around ColPali to facilitate its usage. Feel free to reach out if you want to add your project to this list!

<details>
<summary><strong>🔽 Libraries 📚</strong></summary>

| Library Name  | Description                                                                                                                                                                                                                                          |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
| Byaldi        | [`Byaldi`](https://github.com/AnswerDotAI/byaldi) is [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)'s equivalent for ColPali, leveraging the `colpali-engine` package to facilitate indexing and storing embeddings.                      |
| PyVespa       | [`PyVespa`](https://pyvespa.readthedocs.io/en/latest/examples/colpali-document-retrieval-vision-language-models-cloud.html) allows interaction with [Vespa](https://vespa.ai/), a production-grade vector database, with detailed ColPali support.   |
| Qdrant | Tutorial about using ColQwen2 with the [Qdrant](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/) vector database. |
| Elastic Search     | Tutorial about using ColPali with the [Elastic Search](https://www.elastic.co/search-labs/blog/elastiacsearch-colpali-document-search) vector database. |
| Weaviate | Tutorial about using multi-vector embeddings with the [Weaviate](https://weaviate.io/developers/weaviate/tutorials/multi-vector-embeddings) vector database. |
| Candle        | [Candle](https://github.com/huggingface/candle/tree/main/candle-examples/examples/colpali) enables ColPali inference with an efficient ML framework for Rust.                                                                                        |
| EmbedAnything | [`EmbedAnything`](https://github.com/StarlightSearch/EmbedAnything) Allows end-to-end ColPali inference with both Candle and ONNX backend.                                                                                                           |
| DocAI         | [DocAI](https://github.com/PragmaticMachineLearning/docai) uses ColPali with GPT-4o and Langchain to extract structured information from documents.                                                                                                  |
| VARAG         | [VARAG](https://github.com/adithya-s-k/VARAG) uses ColPali in a vision-only and a hybrid RAG pipeline.                                                                                                                                               |
| ColBERT Live! | [`ColBERT Live!`](https://github.com/jbellis/colbert-live/) enables ColPali usage with vector databases supporting large datasets, compression, and non-vector predicates.                                                                           |
| ColiVara      | [`ColiVara`](https://github.com/tjmlabs/ColiVara/) is retrieval API that allows you to store, search, and retrieve documents based on their visual embedding. It is a web-first implementation of the ColPali paper using ColQwen2 as the LLM model. |
| BentoML       | Deploy ColPali easily with BentoML using [this example repository](https://github.com/bentoml/BentoColPali). BentoML features adaptive batching and zero-copy I/O to minimize overhead.                                                              |
| NoOCR       | NoOCR is end-to-end, [open source](https://github.com/kyryl-opens-ml/no-ocr) solution for complex PDFs, powered by ColPali embeddings. |
| Astra Multi-vector     | [`Astra-multivector`](https://github.com/brian-ogrady/astradb-multivector) provides enterprise-grade integration with AstraDB for late-interaction models like ColPali, ColQwen2, and ColBERT. It implements efficient token pooling and embedding caching strategies to dramatically reduce latency and index size while maintaining retrieval quality. The library leverages Cassandra's distributed architecture for high-throughput vector search at scale. |
| Mixpeek       | [Mixpeek](https://docs.mixpeek.com/processing/feature-extractors) is a production platform for multimodal late-interaction retrieval. It supports models like ColBERT, ColPaLI, and ColQwen2 with built-in indexing, versioning, A/B testing, and explainability across image, text, video, and PDF pipelines. |


</details>

<details>
<summary><strong>🔽 Notebooks 📙</strong></summary>

| Notebook Title                                               | Author & Link                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ColPali Cookbooks                                            | [Tony's Cookbooks (ILLUIN)](https://github.com/tonywu71/colpali-cookbooks) 🙋🏻 |
| Vision RAG Tutorial                                          | [Manu's Vision Rag Tutorial (ILLUIN)](https://github.com/ManuelFay/Tutorials/blob/main/Tuesday_Practical_2_Vision_RAG.ipynb) 🙋🏻 |
| ColPali (Byaldi) + Qwen2-VL for RAG                          | [Merve's Notebook (HuggingFace 🤗)](https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb) |
| Indexing ColPali with Qdrant                                 | [Daniel's Notebook (HuggingFace 🤗)](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html) |
| Weaviate Tutorial                                            | [Connor's ColPali POC (Weaviate)](https://github.com/weaviate/recipes/blob/main/weaviate-features/named-vectors/NamedVectors-ColPali-POC.ipynb) |
| Use ColPali for Multi-Modal Retrieval with Milvus            | [Milvus Documentation](https://milvus.io/docs/use_ColPali_with_milvus.md) |
| Data Generation                                              | [Daniel's Notebook (HuggingFace 🤗)](https://danielvanstrien.xyz/posts/post-with-code/colpali/2024-09-23-generate_colpali_dataset.html) |
| Finance Report Analysis with ColPali and Gemini              | [Jaykumaran (LearnOpenCV)](https://github.com/spmallick/learnopencv/tree/master/Multimodal-RAG-with-ColPali-Gemini) |
| Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs) | [Sergio Paniego](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms) |
| Document Similarity Search with ColPali                      | [Frank Sommers](https://colab.research.google.com/github/fsommers/documentai/blob/main/Document_Similarity_with_ColPali_0_2_2_version.ipynb) |
| End-to-end ColPali inference with EmbedAnything              | [Akshay Ballal (EmbedAnything)](https://colab.research.google.com/drive/1-Eiaw8wMm8I1n69N1uKOHkmpw3yV22w8?usp=sharing) |
| ColiVara: A ColPali Retrieval API                            | [A simple RAG Example](https://github.com/tjmlabs/ColiVara-docs/blob/main/cookbook/RAG.ipynb) |
| Multimodal RAG with Document Retrieval (ColPali), Vision Language Model (ColQwen2) and Amazon Nova | [Suman's Notebook (AWS)](https://github.com/debnsuma/fcc-ai-engineering-aws/blob/main/05-multimodal-rag-with-colpali/01-multimodal-retrival-with-colpali-retreve-gen.ipynb) |
| Multi-vector RAG: Using Weaviate to search a collection of PDF documents | [Weaviate's Notebook](https://github.com/weaviate/recipes/blob/main/weaviate-features/multi-vector/multi-vector-colipali-rag.ipynb) |

</details>

<details>
<summary><strong>🔽 Other resources</strong></summary>

- 📝 = blog post
- 📋 = PDF / slides
- 📹 = video

| Title                                                                                    | Author & Link                                                                                                                                                 |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| State of AI report 2024                                                                  | [Nathan's report](https://www.stateof.ai/) 📋                                                                                                                 |
| Technology Radar Volume 31 (October 2024)                                                | [thoughtworks's report](https://www.thoughtworks.com/radar) 📋                                                                                                |
| LlamaIndex Webinar: ColPali - Efficient Document Retrieval with Vision Language Models   | [LlamaIndex's Youtube video](https://youtu.be/nzcBvba7mzI?si=WL9MsyiAFJMyEolz) 📹                                                                             |
| PDF Retrieval with Vision Language Models                                                | [Jo's blog post #1 (Vespa)](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/) 📝                                                          |
| Scaling ColPali to billions of PDFs with Vespa                                           | [Jo's blog post #2 (Vespa)](https://blog.vespa.ai/scaling-colpali-to-billions/) 📝                                                                            |
| Neural Search Talks: ColPali (with Manuel Faysse)                                        | [Zeta Alpha's Podcast](https://open.spotify.com/episode/2s6ljhd6VQTL2mIU9cFzCb) 📹                                                                            |
| Multimodal Document RAG with Llama 3.2 Vision and ColQwen2                               | [Zain's blog post (Together AI)](https://www.together.ai/blog/multimodal-document-rag-with-llama-3-2-vision-and-colqwen2) 📝                                  |
| ColPali: Document Retrieval with Vision Language Models                                  | [Antaripa Saha](https://antaripasaha.notion.site/ColPali-Efficient-Document-Retrieval-with-Vision-Language-Models-10f5314a5639803d94d0d7ac191bb5b1) 📝        |
| Minimalist diagrams explaining ColPali                                                   | [Leonie's ColPali diagrams on X](https://twitter.com/helloiamleonie/status/1839321865195851859)📝                                                            |
| Multimodal RAG with ColPali and Gemini : Financial Report Analysis Application           | [Jaykumaran's blog post (LearnOpenCV)](https://learnopencv.com/multimodal-rag-with-colpali/) 📝                                                               |
| Implement Multimodal RAG with ColPali and Vision Language Model Groq(Llava) and Qwen2-VL | [Plaban's blog post](https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd) 📝 |
| multimodal AI. open-source. in a nutshell.                                               | [Merve's Youtube video](https://youtu.be/IoGaGfU1CIg?si=yEhxMqJYxvMzGyUm) 📹                                                                                  |
| Remove Complexity from Your RAG Applications                                             | [Kyryl's blog post (KOML)](https://kyrylai.com/2024/09/09/remove-complexity-from-your-rag-applications/) 📝                                                   |
| Late interaction & efficient Multi-modal retrievers need more than a vector index        | [Ayush Chaurasia (LanceDB)](https://blog.lancedb.com/late-interaction-efficient-multi-modal-retrievers-need-more-than-just-a-vector-index/) 📝                |
| Optimizing Document Retrieval with ColPali and Qdrant's Binary Quantization              | [Sabrina Aquino (Qdrant)]( https://youtu.be/_A90A-grwIc?si=MS5RV17D6sgirCRm)  📹                                                                              |
| Hands-On Multimodal Retrieval and Interpretability (ColQwen + Vespa)                     | [Antaripa Saha](https://www.analyticsvidhya.com/blog/2024/10/multimodal-retrieval-with-colqwen-vespa/) 📝                                                     |

</details>

## Citation

**ModernVBERT: Towards Smaller Visual Document Retrievers**

```latex
@misc{teiletche2025modernvberttowardssmallervisual,
      title={ModernVBERT: Towards Smaller Visual Document Retrievers}, 
      author={Paul Teiletche and Quentin Macé and Max Conti and Antonio Loison and Gautier Viaud and Pierre Colombo and Manuel Faysse},
      year={2025},
      eprint={2510.01149},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/html/2510.01149v1}, 
}
```

**ColPali: Efficient Document Retrieval with Vision Language Models**

```latex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}
```

**ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval**

```latex
@misc{macé2025vidorebenchmarkv2raising,
      title={ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval}, 
      author={Quentin Macé and António Loison and Manuel Faysse},
      year={2025},
      eprint={2505.17166},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.17166}, 
}
```
