# COS760 Natural Language Research Project

## Local Development:
To get started clone the repository:
```shell
# With https:
git clone https://github.com/GremBleen/COS-760-Project.git

# With ssh:
git clone git@github.com:GremBleen/COS-760-Project.git
```

After the repository is available locally, it is necessary to create a `.env` file. This is where all environment variables and is necessary so as not to expose sensitive information.

The `.env` file requires the following:
```env
HUGGINGFACE_TOKEN= <INPUT TOKEN HERE>
```

Now that you have the basic project structure, you can start looking at the python setup.

>**Note:**  The project made use of conda for the management of the environment and dependencies.

An `environemnt.yml` has been included from which a conda environment with all necessary dependencies can be created.

To create the environment, ensure that conda is installed and run the following command:
```shell
conda env create -f environment.yaml
```

Now you should be able to activate the environment:
```shell
conda activate cos-760-env
```

Now to run the project, you can use the following command (assuming you are in the root directory of the project):
```shell
python3 src/main.py
```

>**Note:** If more dependencies are required, it is necessary to regenerate the `environment.yml` file. This can be done as follows:
```shell
# Linux or MacOS
conda env export | sed '/^prefix:/d' > environment.yml

# Windows
conda env export | Select-String -NotMatch "^prefix:" > environment.yml
```