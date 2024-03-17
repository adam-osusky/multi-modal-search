# Simple multi modal search and RAG
Lightweigh multimodal RAG for single pdf document with local LLM llava. Chat with your pdf! This project is for educative purposes. For now it is only a command line app.

## Getting Started
TODO

It is a poetry python project for which you need some system dependencies. Specifically
- Tesseract OCR
- opnencv
- poppler
- ollama
  
Look at the `Dockerfile` file for inspiration on how to download them.

### Poetry Installation
TODO
### Docker
Because this projects uses Large Language Model locally you want to have GPU support. At the moment this projects supports only nvidia gpus. Make sure you have installed: [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

1. Clone the repo
```bash
git clone https://github.com/adam-osusky/multi-modal-search.git
cd multi-modal-search
```

2. Build the image. This can take a while.
```bash
docker build -t mulmod .
```

3. Start shell in the container.
```bash
docker run -it mulmod:latest /bin/bash
```

4. Download LLM model from ollama.
```bash
nohup bash -c "ollama serve &" && sleep 7 && ollama pull llava
```

5. Now you can download pdf that you want to chat with and start the main.
```bash
!curl -o llm.pdf https://arxiv.org/pdf/2307.06435.pdf
python3 src/mulmod/main.py llm.pdf 1
```


## Usage
