---
layout: post
title: Setting up Kaggle API on Mac
---

Hello there!

If you are like me and want to use Kaggle API instead of manual clicks here and there on the Kaggle website to get your task done, this post is for you!

Let's get started!

**1. Installing the library**

There are two ways to do this:
- Using the command `pip install kaggle`
- Using the command `pip install --user kaggle`

*Note: Avoid using `sudo pip install kaggle` if you don't want to face unnecessary problems or unless you know what you're doing.*

**2. Setting up the API token**

- Go to the [kaggle website](https://www.kaggle.com).
- Click on `Your profile` button on the top right and then select `My Account`.
- Scroll down to the `API` section and click on the `Create New API Token` button.
- It will initiate the download of a file call `kaggle.json`. Save the file at a known location on your machine.

**3. place the `.json` file at the correct location**

- Move the downloaded file to a location `~/.kaggle/kaggle.json`. If you don't have the .kaggle folder in your home directory, you can create one using the command:
  - `mkdir ~/.kaggle`
- Now move the downloaded file to this location using:
  - `mv <location>/kaggle.json ~/.kaggle/kaggle.json`
- You need to give proper permissions to the file (since this is a hidden folder):
  - `chmod 600 ~/.kaggle/kaggle.json`

**4. Checking if it works**

- Run the command `kaggle competitions list`.
- If you see a list of active competitions, you're done setting the API up.
- If you don't, try looking at the location of kaggle by using the command:
  - `pip uninstall kaggle`

  This will ask you to confirm whether you want to uninstall by telling you the location of kaggle on your machine.
- So if you find the location such as `~/.local/bin/kaggle`, try running the kaggle command as:
  - `~/.local/bin/kaggle competitions list`

  And it should work this time. You can actually export this binary path to the environment if you don't want to type the whole path again and again.

**5. Using the API**

- Now that you've set up the API, you can start using it. You can get the help for the api using `kaggle --help`. If you want help regarding a more specific command use `--help` after that command. For example, `kaggle dataset --help`. For a more detailed description of everything and other information, go to the [kaggle-api GitHub repository](https://github.com/Kaggle/kaggle-api).

Sources:
1. [kaggle-api GitHub repository](https://github.com/Kaggle/kaggle-api)
