# ML basics: map of the field and learning path

## Which part of IT is this?

**Machine learning (ML)** is the umbrella, and most of the topics below fall under its subfield **deep learning**.

### 1. Recognizing images, text, voice
Application domains of ML, each with its own name:
- **Computer vision (CV):** image classification, object detection, segmentation, OCR
- **Natural language processing (NLP):** text understanding and generation, including LLMs
- **Speech/audio processing:** speech recognition (ASR), text-to-speech, speaker identification

Modern systems in all three are mostly deep neural networks (CNNs, transformers).

### 2. Building and using AI models
- **ML/deep learning engineering:** designing architectures and writing model code (PyTorch, TensorFlow/Keras, JAX)
- **Applied AI / AI engineering:** using pretrained models via APIs or libraries (Hugging Face, OpenAI/Anthropic APIs, RAG, fine-tuning) without necessarily training from scratch

### 3. Models for cloud and local (PC, mobile, IoT)
- **MLOps / ML systems:** deploying, serving, monitoring, and versioning models in production
- **Edge ML / TinyML:** running models on constrained devices, using quantization, pruning, and distillation, with runtimes like ONNX Runtime, TensorFlow Lite, Core ML, ExecuTorch, llama.cpp, and OpenVINO

### 4. Model training
- **Core ML/deep learning:** optimization, loss functions, regularization, data pipelines, evaluation
- **Distributed training and ML infrastructure:** GPUs/TPUs, mixed precision, data/model parallelism
- **Data engineering:** dataset collection, labeling, cleaning

## Roles you'll see in job listings
- **ML engineer:** training and building models
- **MLOps engineer:** deployment and infrastructure
- **AI/applied engineer:** integrating models into products
- **Research scientist:** new methods
- **Edge/embedded ML engineer:** models on constrained devices

## Suggested path for an experienced developer
1. ML fundamentals and Python stack: NumPy, scikit-learn, then PyTorch
2. Deep learning basics: backprop, CNNs, transformers (fast.ai or Karpathy's "Zero to Hero" videos are good, practical starts)
3. Pick one domain (vision, NLP, or speech) and build a small project end to end
4. Then deployment: export to ONNX, quantize, run on a PC and a phone or Raspberry Pi
5. Optionally MLOps tooling: MLflow, Docker/Kubernetes, model serving

## Resources
- [fast.ai: Making neural nets uncool again](https://www.fast.ai/)
- [Andrej Karpathy: The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (YouTube, "Neural Networks: Zero to Hero" playlist)
- [ ML Foundations for AI Engineers (in 34 Minutes)](https://youtu.be/BUTjcAjfMgY?si=fuoaU7oZMJZTO_YO)