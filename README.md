# Mini GPT
*Mostly a learning exercise to reproduce GPT-2 and take a stab at code synthesis from scratch.* 

## File Notes
---
1. `tiny.py` simplest model implementation using top level API.
2. `train.py` simplest training loop using lightning ⚡.
3. `basic.ipynb` end to end demo of usage.





# General Note

### Useful Links
* https://github.com/karpathy/minGPT
* https://www.kaggle.com/code/shashwatwork/mini-gpt-experiment-karpathy
* https://jaykmody.com/blog/gpt-from-scratch/
* https://www.youtube.com/watch?v=kCc8FmEb1nY
* https://www.youtube.com/watch?v=d7IRM40VMYM
* https://github.com/Lightning-AI/lightning-GPT
* https://openai.com/blog/introducing-chatgpt-and-whisper-apis

### Implementation Guides
*  [https://blog.paperspace.com/sentence-embeddings-pytorch-lightning/](https://blog.paperspace.com/sentence-embeddings-pytorch-lightning/)
* [https://sachinruk.github.io/blog/deep-learning/2022/09/25/grammar-correction-via-gpt2.html](https://sachinruk.github.io/blog/deep-learning/2022/09/25/grammar-correction-via-gpt2.html)
* [https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial)
* [https://course.fast.ai/](https://course.fast.ai/)

### Goals
* [ ] Fix GPU integration
* [ ] Check that VAE is properly encoding/decoding
* [x] Implement core GPT-2 architecture 
  * Extend this to the following:
    * GPU integration
    * LLVM compilation & training
    * Text generation
    * [ ] Complete loss fn
    * [ ] Add generator function (512 tokens)
* [ ] Make a python autocompletion engine
  * [ ] Hook into VSCode as an extension
  * [ ] Follow evaluation up with [CodeGen](https://github.com/salesforce/CodeGen)
* [ ] Enable resumable training w/ checkpoints
* [ ] Integrate with https://pre-commit.com/

* [ ] Investigate tooling like the following:
  * https://github.com/hwchase17/langchain
  * https://primo.ai/index.php?title=Generative_Pre-trained_Transformer_(GPT)
  * https://gpt-index.readthedocs.io/en/latest/getting_started/starter_example.html


### Implementation Phases
1. Pair program with GPT
2. Follow guides
3. License taggging

