- Quickly download all checkpoints from [Model_Zoo](https://huggingface.co/AaronCIH/RAR_modelzoo), or download individual checkpoints as needed.
```
cd checkpoints/
python download.py
```
    - [CLIP-ViT-L-14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt). Required. 
    - [Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5). Required. 
    - [All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Required only for confidence estimation of detailed reasoning responses. 
    - [LQA](https://huggingface.co/AaronCIH/RAR_modelzoo/tree/main/LQA) 
    - [AaronCIH/SD35_IR](https://huggingface.co/AaronCIH/SD35_IR/tree/main) 
    - [RAR](https://huggingface.co/AaronCIH/RAR_modelzoo/tree/main/RAR)