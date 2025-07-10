#  Multi-Label Ingredient Classification with BERT

 **Colab Notebook Link**: [Open in Google Colab](https://colab.research.google.com/drive/1KJmsDEDuxPUk_R7l76fENbQ7LLtXUgrR?usp=sharing)


#Project Overview

This project focuses on **multi-label text classification** using a pre-trained BERT model. 
The goal is to identify the presence of the **top 30 ingredients** in cocktail recipes based on the recipe text description. 
This task fine-tunes a BERT model to perform multi-label classification, demonstrating how transformer models can be adapted to custom NLP tasks.


#Dataset Description

**Source**: Cocktail recipes CSV (custom dataset from `/MyDrive/LLM ASSIGNMENT/cocktails_recipe.csv`)
**Fields Used**:
`recipe`: Description or steps of cocktail preparation.
`ingredients`: Comma-separated list of ingredients.

*Cleaning & Preprocessing*:
Removed rows with missing values.
Extracted and parsed ingredients.
Selected the **top 30 most frequent ingredients** for classification.
Created binary multi-label vectors for each recipe based on ingredient presence.



#Model & Methodology

**Model**: `BertForSequenceClassification` from Hugging Face Transformers.
**Tokenizer**: `BertTokenizer` (bert-base-uncased).
**Training Strategy**:
Fine-tuned for **multi-label classification** using `BCEWithLogitsLoss`.
Used a custom `Trainer` class to handle multi-label outputs.
Evaluation metrics included `f1_micro`, `f1_macro`, `precision`, `recall`, and `hamming_loss`.



#Evaluation Metrics

After training the model for 3 epochs, the evaluation was performed using the test set.

| Metric          | Value  |
|------------------|--------|
| F1 Micro         | ~0.71  |
| F1 Macro         | ~0.52  |
| Hamming Loss     | ~0.15  |
| Precision        | ~0.68  |
| Recall           | ~0.72  |

( Values may vary slightly depending on the runtime execution. )



#Visualizations

Several plots were used to analyze the dataset:

 **Histogram of number of ingredients per recipe**
 **Top 30 ingredients by frequency**
 **Recipe length distribution (in words)**
 **Heatmap showing ingredient co-occurrence correlation**

These helped better understand the structure and features of the dataset before training.


