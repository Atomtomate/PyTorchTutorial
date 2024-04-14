# Overview

## Context

This repository was created as part of a tutorial for getting started with machine learning in general and `PyTorch` with `PyTorch Lightning` in specific.
As such, the written documentation may be lacking in parts.

## Structure

We will first go over the Fundamentals of supervised learning and then apply those to some simple examples, using PyTorch.
After that, we will move on to PyTorch Lightning and experiment with more advanced models, also introducing more tools like Tensorboard and the Learning Interpretability Tool.
Using this more convenient setup, we will explore the usual workflow of developing, refining (simple) machine learning models and interpreting their behavior.
My goal is to give you enough hands-on experience to be able to start your own projects right away.
During this process, I will introduce common phrases and explain them.

## Directory structure

# Setup

You will need a Python environment with a lot of additional packages.
In order to avoid conflicts with other projects, one usually creates environments for each of them.
Here we will use [Conda](https://conda.io/projects/conda/en/latest/index.html).

After installing Conda we:
 - [create an environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for this project using `conda create --name MLTut` (you can of cause change the name).
 - activate the environment with `activate MLTut` (it should be active from the start, but you may have to do this after reboots)
 - install packages using `pip install <name>`. For now, you can try `pip install ipython lightning lightning-utilities matplotlib numpy torch pytorch-lightning torchvision scipy tensorboard torchview graphviz`.

For the live tutorial, we will use [VScode](https://code.visualstudio.com/Download).

After installation, we need to setup a couple of things and [install extensions](https://code.visualstudio.com/docs/editor/extension-marketplace).
You can easily search and install all plugins from the tab inside VSCode, in case you run into problems I will provide links here anyway:

 - [Live Share](https://code.visualstudio.com/learn/collaboration/live-share)
 - [Python](https://code.visualstudio.com/docs/languages/python)
 - [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
 - Tensorboard

If you installed VSCode after conda, you may need to point it to the correct Python executable. 
Go to File > Preferences > Settings and search for `Default Interpreter`. 
This should point you towards the setting for the Python plugin where you can specify the path to the executable from the environment above.
you can obtain the path from the console by activating the conda environment (see above) and typing `which python` under Linux or `where python` under Windows.
Make sure to use the Python executable from a path having the `MLTut` (or the name you chose before) in it.
Paste this path into the Python plugin field for Default Interpreter.

## Getting help

Since machine learning and Python have developed to a point where people outside the computer science community are using these tools regularly, the amount of forum posts and tutorials for all kinds of basic questions is very large.
You may also find that LLMs like ChatGPT are quite helpful with questions regarding the setup and standard error codes.
Also, feel free to contact me or open an issue, if you want to reuse this tutorial.

# Credits

My setup for PyTorch Lightning has been adapted from [Daniel Springer](https://github.com/DanielSpringer/LuttingerWard_from_ML) and [Marco Knipfer](https://github.com/BoGGoG/PyTorchLightningTemplate) with many helpful comments from both.