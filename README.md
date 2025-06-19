# COS760 Natural Language Research Project

This repository contains the code for the COS760 Natural Language Research Project, which focuses on ASR models for African languages. The project is part of the COS-760 module provided by the University of Pretoria. 

This project was developed by the following team members:
- [Graeme Blain](https://github.com/GremBleen)
- [Aidan Chapman](https://github.com/Acedem)
- [Matjere Matseba](https://github.com/MatjereJ)

## Local Development

To get started, clone the repository:

```shell
# With HTTPS:
git clone https://github.com/GremBleen/COS-760-Project.git

# With SSH:
git clone git@github.com:GremBleen/COS-760-Project.git
```

## Python Setup

After cloning the repository, you can set up the Python environment.

### Python Version

We use Python 3.11 for this project. It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects. You can use `venv`, `pipenv`, `conda`, or any environment manager of your choice. Below are some options to create a virtual environment:

```shell
python3 -m venv venv
```

If you have `pipenv` installed:

```shell
pipenv --python 3.11
```

If you are using conda:

```shell
conda create -n venv python=3.11
```

### Installing Dependencies

After setting up the virtual environment, activate it and install the required dependencies.

Activate the virtual environment:

```shell
source venv/bin/activate  # For Linux/MacOS
```
```shell
venv\Scripts\activate  # For Windows
```

> **Note:**  
> The project originally used conda for environment and dependency management. However, due to issues when creating environments on different operating systems, we now recommend installing dependencies manually. If you encounter errors when running the project, check if additional dependencies are required.

Below is a list of required dependencies:

```
datasets>=3.6.0
dotenv>=0.9.9
fuzzywuzzy>=0.18.0
huggingface-hub>=0.33.0
jiwer>=3.1.0
librosa>=0.11.0
matplotlib>=3.10.3
python-levenshtein>=0.27.1
torch>=2.7.1
torchaudio>=2.7.1
transformers>=4.52.4
```

## Running the Models

After completing the setup, create a `.env` file in the root directory. This file stores environment variables and helps keep sensitive information secure.

The `.env` file should contain:

```env
HUGGINGFACE_TOKEN=<INPUT TOKEN HERE>
```

To run the models, create a `presets.json` file in the main directory with the following structure:

```json
{
    "dataset_language": "<LANGUAGE CODE>",
    "model": "<MODEL NAME>",
    "batch_size": 20,
    "refinement_method": true,
    "debug": true
}
```

Replace `<LANGUAGE CODE>` with the code for your dataset's language (e.g., "afr" for Afrikaans, "xho" for Xhosa, "zul" for Zulu).  
Replace `<MODEL NAME>` with the desired model (e.g., "facebook-mms", "lelapa", "sm4t", "wav2vec", "whisper-large", "whisper-medium").  
You can adjust `batch_size`, but we recommend keeping it at 20 for a good balance between performance and memory usage.  
Set `refinement_method` to `true` or `false` depending on whether you want to use the refinement method.  
Set `debug` to `true` or `false` to control whether running information is printed to the console.

### Running the Project

To run the project, use the following command from the root directory:

```shell
python3 src/main.py
```

## Git Development

### Git Workflow

We use the [Git Flow](https://danielkummer.github.io/git-flow-cheatsheet/) workflow.

If Git Flow is installed, initialize it with:

```shell
git flow init
```

To start a new feature:

```shell
git flow feature start <feature-name>
```

To finish a feature:

```shell
git flow feature finish <feature-name>