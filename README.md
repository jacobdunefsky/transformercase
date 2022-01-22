# transformercase

The code for my part of my final project for LING 380: Topics in Computational Linguistics: Neural Network Models of Linguistic Structure. [For the paper, click here.](https://raw.githubusercontent.com/jacobdunefsky/transformercase/main/EmbeddingsOfCzechNouns.pdf)

---

## Disclaimer

These scripts were used to record, analyze, and plot data to be used in the paper. As such, these scripts were used interactively from the Python REPL on the fly. Similarly, they were haphazardly iterated upon, as new experiments were devised. For these reasons, the code isn't exactly the best-written; please pardon the dust.

## Usage

The dataset of Czech noun forms by case is stored as a Python pickle file as `cases.pkl`.

`finetune_transformer.ipynb` allows the transformer to be finetuned on the dataset of Czech nouns.

`default_case_var.py` calculates case variances for the pretrained transformer.

`finetuned_case_var.py` calculates case variances for the finetuned transformer.

`fasttext_case_var.py` calculates case variances for the FastText model.

`plot_pretrained_data.py` plots the data from the pretrained transformer and reports statistics.

`plot_finetuned_data.py` plots the data from the finetuned transformer and reports statistics.

`plot_fasttext_data.py` reports statistics from the FastText model (no plotting is necessary, because there is only one "layer" in the FastText model).
