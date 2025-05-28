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

Initially we had an `environment.yml` file, however, there were issues when creating a new environment on a different OS, so instead we recommend that you go through the python files and install the dependencies manually. When running the project, if any errors occur, look to see if further dependencies are necessary.

Now to run the project, you can use the following command (assuming you are in the root directory of the project):
```shell
python3 src/main.py
```

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