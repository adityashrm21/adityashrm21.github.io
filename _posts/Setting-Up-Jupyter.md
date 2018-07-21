---
layout: post
title: Installing and Setting Up Jupyter Notebook on Macbook
---

Here are the steps you need to follow in order to set up Jupyter Notebook on your Macbook:

1. Go to the Anaconda website and download the installer package for Mac OS:
https://www.anaconda.com/download/#macos

2. After the download completes, run the installer and follow the steps in order to install Anaconda on your machine.

3. Now open your terminal and try to run the command:
>jupyter notebook

You will get an error saying:
* jupyter : command not found

This is because you need to add the anaconda path to your bash profile.

4. Open the bash profile using the command:
>vi .bash_profile

5. The file must be having content similar to:
> added by Anaconda3 5.2.0 installer
>
>export PATH="/anaconda3/bin:$PATH"

6. Change it to:
> added by Anaconda3 5.2.0 installer
>
>export PATH="$PATH:/anaconda3/bin"

7. Restart your terminal
8. Try running the jupyter notebook again:
>jupyter notebook

9. The server should run on **localhost:8888**.
