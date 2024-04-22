babyLM Challenge
==============================

**Deadlines**

| Task                 | Deadline                                   |
|----------------------|--------------------------------------------|
| Results submission   | September 13, 23:59 anywhere on earth (UTC-12) |
| Paper submission     | September 20, 23:59 anywhere on earth (UTC-12) |


**Overview**

Huge effort has been put into optimizing LM pretraining at massive scales in the last several years. While growing parameter counts often get the most attention, datasets have also grown by orders of magnitude. For example, Chinchilla sees 1.4 trillion words during training---well over 10000 words for every one word a 13 year old child has heard in their entire life. 

The goal of this shared task is to incentivize researchers with an interest in pretraining or cognitive modeling to focus their efforts on optimizing pretraining given data limitations inspired by human development. Additionally, we hope to democratize research on pretraining—which is typically thought to be practical only for large industry groups—by drawing attention to open problems that can be addressed on a university budget.

**Submission Tracks**

- There are four competition categories: multimodal, strict, strict-small and paper track.

- We will focus on Strict and Strict-Small Tracks: The strict and strict-small tracks require that submissions are trained on 100M words (for strict) or 10M words (for strict small) of text data. 

**Pretraining data**

- Text-only Dataset:  text-only dataset is an updated version of 2023's BabyLM training corpus. It comes in 10M and 100M word variants, consists mostly of transcribed speech, and has a large proportion of simplified language, such as child-directed speech, childrens' storybooks, and simple Wikipedia.

Contents of the text-only dataset
- `babylm_100M`: 100M-word training set for the *strict* track.
- `babylm_10M`: 10M-word training set for the *strict-small* track.
- `babylm_dev`: Development set for both tracks (10M words)
- `babylm_test`: Test set for both tracks (10M words)

**Composition of the data**

All datasets are sampled from a mixture of 6 data domains, shown below, along with their respective weights in the distributed dataset.

| Source | Weight | Domain | Citation | Website | License |
| --- | --- | --- | --- | --- | --- |
| BNC | 8% | Dialogue | BNC Consortium (2007) | [link](http://www.natcorp.ox.ac.uk/) | [link](http://www.natcorp.ox.ac.uk/docs/licence.html) <sup>1</sup> |
| CHILDES | 29% | Dialogue, Child-Directed | MacWhinney (2000) | | [link](https://talkbank.org/share/rules.html) |
| Project Gutenberg | 26% | Fiction, Nonfiction | Gerlach & Font-Clos (2020) | [link](https://github.com/pgcorpus/gutenberg) | [link](https://www.gutenberg.org/policy/license.html) |
| OpenSubtitles | 20% | Dialogue, Scripted | Lison & Tiedermann (2016) | [link](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Open source |
| Simple English Wikipedia | 15% | Nonfiction | -- | [link](https://dumps.wikimedia.org/simplewiki/20221201/) | [link](https://dumps.wikimedia.org/legal.html) |
| Switchboard | 1% | Dialogue | Godfrey et al. (1992), Stolcke et al., (2000) | [link](http://compprag.christopherpotts.net/swda.html) | [link](http://compprag.christopherpotts.net/swda.html) |

**Data preprocessing**



Implementation details
------------

>Insert table summary on training details 


Setup
------------

**Create environment and install dependencies**

For macOS (Apple Chip):
```bash
$ mamba env create -f dependencies/babylm-conda-metal.yaml
```

**Download data**

[Click here to download dataset](https://osf.io/ad7qg/) and save the dev, text, train_10M and train_100M to the `data/raw` folder



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
