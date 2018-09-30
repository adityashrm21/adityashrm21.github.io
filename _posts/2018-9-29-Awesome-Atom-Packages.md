---
layout: post
title: Awesome Atom Packages
---

If you are an Open Source enthusiast like me and are in love with Atom, you definitely would want to integrate a lot of useful packages and functionalities in Atom to make your life easier!

In this post, I will write about some of the most useful packages that I have installed.

Let's get started!

**1.** The first one to join this list and which inspired me to write this post in the first place is the [git-plus](https://atom.io/packages/git-plus) package.
- This is a super-useful package for someone who works mostly on atom and doesn't like typing git commands or prefer git-clients.
- One thing to note:
**Make sure your `gitconfig` file is configured. You must configure at least the `user.email` and `user.name` variables.**
- The usage of the package is really easy once you install and set it up! Whenever you add a project folder initiated with a .git file, you will be able to see which files have been committed, which are newly added and which are modified in different colors in the `tree view`.
- You can use either of these options for showing the Git-Plus Palette:
  - Cmd-Shift-H on MacOS
  - Ctrl-Shift-H on Windows + Linux
  - Git Plus: Menu on the atom command palette.
- You can also right click on any file or project folder and select `Git` option from the menu and choose the operation you want to perform on the file or folder.
- You will definitely need to type in the commit message though (no escape from that!).
- After typing in the message just save the COMMIT_EDITMSG file that pops up and a `git commit` command will be issued.
- To push the changes to remote repository, just choose `Git push` from the Git submenu.
- Pulling any changes is also as simple as that - Just select the `Git pull` from the submenu. Of course, the other commands can also be directly used from the menu.
- I found this package quite useful and I am definitely going to dig more into it! Hope you like it too.
