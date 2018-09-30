---
layout: post
title: Installing Python3 on Macbook
---

Here are the simple steps needed to be followed in order to install python3 on your Macbook:

1. First we will install Homebrew, which is a free and open-source software package management system that simplifies the installation of software on Apple's macOS operating system. Copy and paste the following command in your terminal and run it:
    ```bash
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    ```
2. Follow the instructions if any and after the command finishes running, brew would be installed on your system. Now you can install many packages and software using brew easily. By default your Mac should contain a python version which you can check by using:

    ```bash
    python --version
    ```
3. For me it gives "Python 2.7.10". Now to install python3 simply use the command:

    ```bash
    brew install python3
    ````
4. And you're done!
