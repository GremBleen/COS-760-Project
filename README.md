# COS760 Natural Language Research Project

## Local Development:
To get started clone the repository:
```shell
# With https:
git clone https://github.com/GremBleen/COS-760-Project.git

# With ssh:
git clone git@github.com:GremBleen/COS-760-Project.git
```

## Python Setup:

Now that you have the basic project structure, you can start looking at the python setup.

### Python Version:
We are using Python 3.11 for this project. It is recommended to use a virtual environment to manage the dependencies and avoid conflicts with other projects. The recommended way to set up a virtual environment is to use `pipenv`, `conda` or any environment manager. You can create a virtual environment with the following command:

```shell
python3 -m venv venv
```

or if you have `pipenv` installed:
```shell
pipenv --python 3.11
```

or if you are using conda, you can create an environment with:
```shell
conda create -n venv python=3.11
```

### Installing Dependencies:
After setting up the virtual environment, you need to install the dependencies required for the project. You can do this by activating the virtual environment and then installing the dependencies using `pip` or `pipenv`.

Activate the virtual environment

```shell
source venv/bin/activate  # For Linux/MacOS
```
```shell
venv\Scripts\activate  # For Windows
```
Then install the dependencies

>**Note:**  The project made use of conda for the management of the environment and dependencies.

Initially we had an `environment.yml` file, however, there were issues when creating a new environment on a different OS, so instead we recommend that you install the dependencies manually. When running the project, if any errors occur, look to see if further dependencies are necessary.

Below is a list of the dependencies that are required for the project:

```
dependencies = [
    "datasets>=3.6.0",
    "dotenv>=0.9.9",
    "fuzzywuzzy>=0.18.0",
    "huggingface-hub>=0.33.0",
    "jiwer>=3.1.0",
    "librosa>=0.11.0",
    "matplotlib>=3.10.3",
    "python-levenshtein>=0.27.1",
    "torch>=2.7.1",
    "torchaudio>=2.7.1",
    "transformers>=4.52.4",
]
```

## Running the Models:

After all the setup is, it is necessary to create a `.env` file. This is where all environment variables and is necessary so as not to expose sensitive information.

The `.env` file requires the following:
```env
HUGGINGFACE_TOKEN= <INPUT TOKEN HERE>
```

To run the models, you must create a `presets.json` file in the main directory of the project. This file should contain the following structure:

```json
{
    "dataset_language": "<LANGUAGE CODE>",
    "model": "<MODEL NAME>",
    "batch_size": 20,
    "refinement_method": true,
    "debug": true
}
```

In the above structure, replace `<LANGUAGE CODE>` with the language code of the dataset you want to use (e.g., "afr" for Afrikaans, "xho" for Xhosa, and "zul" for Zulu). Also replace `<MODEL NAME>` with the name of the model you want to use (e.g., "facebook-mms", "lelapa", "sm4t", "wav2vec", "whisper-large", and "whisper-medium"). The `batch_size` can be adjuste, but we recommend keeping it at 20, as it provides a good balance between performance and memory usage. The `refinement_method` can be set to `true` or `false`, depending on whether you want to use the refinement method or not. The `debug` flag can also be set to `true` or `false`, depending on whether you want to print running information to the console or not.

### Running the Project:

Now to run the project, you can use the following command (assuming you are in the root directory of the project):
```shell
python3 src/main.py
```

## Git Development:

### Git Workflow:
We are making use of the [Git Flow](https://danielkummer.github.io/git-flow-cheatsheet/) workflow.

Assuming git flow has been installed, you can initialize it as follows:
```shell
git flow init
```

To start a new feature, you can use the following command:
```shell
git flow feature start <feature-name>
```

To finish a feature, you can use the following command:
```shell
git flow feature finish <feature-name>
```