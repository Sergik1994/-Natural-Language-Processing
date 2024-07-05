# Natural Language Processing

## Project

Elbrus Bootcamp | Phase-2 | Team Project

## 🦸‍♂️ Team
1. [Sergey Nadanyan](https://github.com/Sergik1994)
2. [Nickolay Zefirov](https://github.com/kolaz92)
3. [Savr Ovalov](https://github.com/SavrOverSide)

## 🎯 Task    
Development of a multipage application using streamlit [(Streamlit service deployed on HuggingFace Spaces)](https://huggingface.co/spaces/AntNikYab/NaturalLanguageProcessing):

- Page 1 • Review Classification for Polyclinics

        User-entered review classification model
        Outputs prediction results by three models:
        1) Classic ML algorithm (Logistic Regression) trained on BagOfWords representation
        2) LSTM model
        3) BERT-based
        Alongside the prediction, the time it was obtained is displayed
        The page features a comparative table based on the f1-macro metric for all constructed classifiers

- Page 2 • Text Generation with GPT Model

        Generative model 
        Users can adjust the length of the generated sequence
        The number of generations, temperature, top-k/p, maximum length, and the number of generated sequences can be controlled
