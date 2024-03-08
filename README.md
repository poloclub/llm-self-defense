# LLM Self Defense

[![BSD3 license](http://img.shields.io/badge/license-BSD3-brightgreen.svg)](https://opensource.org/license/bsd-3-clause)
[![arXiv](https://img.shields.io/badge/arXiv-2308.07308-b3131b.svg)](https://arxiv.org/abs/2308.07308)

[LLM Self Defense: By Self Examination, LLMs know they are being tricked](https://arxiv.org/abs/2308.07308). Mansi Phute, Alec Heibling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau. In *ICLR 2024 TinyPaper*, 2024.

📄  <a href="https://arxiv.org/abs/2308.07308"> Research Paper</a> &nbsp;&nbsp;&nbsp;&nbsp;
🚀  <a href="https://mphute.github.io/papers/llm-self-defense"> Project Page</a> &nbsp;&nbsp;&nbsp;&nbsp; 

<p align="center">
    <img src="imgs/crown-jewel.png" alt="drawing" width="60%"/>
</p>

Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM SELF DEFENSE, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM SELF DEFENSE on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM SELF DEFENSE succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.

## News

- 📄  Accepted to ICLR TinyPapers 2024
- ⭐ Highlighted in ACL 2024 Tutorial: [*Vulnerabilities of Large Language Models to Adversarial Attacks*](https://llm-vulnerability.github.io)
- 🚀 Deployed at ADP, The largest payroll company in the world

## Getting Started

### Environment Setup

Create a virtual envirmonemt witht he following command 

```conda create --name llmdefense python=3.9 --file requirements.txt```


### API Keys and logins

In ```harm_filter.py``` if you are using GPT 3.5 add the  ```OPENAI_API_KEY``` on line 71

You might also need to login to your huggingface account to access Llama weights using the following command

```huggingface-cli login```

### Evaluation

Run the following command

```python3 harm_filter.py```

You can test your own data by modifying the ```DATA_PATH``` variable, and change the model used as harm filter by changing the ```HARMFILTER_MODEL``` 



## Citation
```bibtex
@article{phute2023llm,
  title={Llm self defense: By self examination, llms know they are being tricked},
  author={Phute, Mansi and Helbling, Alec and Hull, Matthew and Peng,ShengYun and Szyller,Sebastian and Cornelius,Cory and Chau, Duen Horng},
  journal={arXiv preprint arXiv:2308.07308},
  year={2023}
}
```

## Contact
If you have any questions, feel free to open an issue or contact [Mansi Phute](https://mphute.github.io) (CS MS @Georgia Tech).