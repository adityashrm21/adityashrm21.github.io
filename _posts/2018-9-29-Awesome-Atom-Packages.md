---
layout: post
title: Awesome Atom Packages
---

If you are an Open Source enthusiast like me and are in love with Atom, you definitely would want to integrate a lot of useful packages and functionalities in Atom to make your life easier!

In this post, I will write about some of the most useful packages that I have installed.

Let's get started!

### git-plus
The first one to join this list and which inspired me to write this post in the first place is the [git-plus](https://atom.io/packages/git-plus) package.
- This is a super-useful package for someone who works mostly on atom and doesn't like typing git commands or prefer git-clients.<br><br>
- One thing to note:
**Make sure your `gitconfig` file is configured. You must configure at least the `user.email` and `user.name` variables.**<br><br>
- The usage of the package is really easy once you install and set it up! Whenever you add a project folder initiated with a .git file, you will be able to see which files have been committed, which are newly added and which are modified in different colors in the `tree view`.<br><br>
- You can use either of these options for showing the Git-Plus Palette:
  - Cmd-Shift-H on MacOS
  - Ctrl-Shift-H on Windows + Linux
  - Git Plus: Menu on the atom command palette.<br><br>
- You can also right click on any file or project folder and select `Git` option from the menu and choose the operation you want to perform on the file or folder.<br><br>
- If you are choosing the `Git commit` option, you will definitely need to type in the commit message though (no escape from that!).<br><br>
- After typing in the message just save the COMMIT_EDITMSG file that pops up and a `git commit` command will be issued.<br><br>
- To push the changes to remote repository, just choose `Git push` from the Git submenu.<br><br>
- Pulling any changes is also as simple as that - Just select the `Git pull` from the submenu. Of course, the other commands can also be directly used from the menu.<br><br>
- I found this package quite useful and I am definitely going to dig more into it! Hope you like it too.

### markdown-preview-plus

[This package](https://atom.io/packages/markdown-preview-plus) is super useful for people who use markdown to write technical blogs, documentation and reports.

I use Jupyter Notebooks and I find the idea of writing a `.ipynb` file to create a markdown report really frustrating and I didn't want to use any third party markdown editors specifically for this purpose. I have tried using [boostnote.io](https://boostnote.io/) and it is a good markdown editor but I don't prefer using a separate editor only for markdown. If you like it, go ahead with it!

- Though atom let's you preview markdown by default (I guess), it is not quite helpful. The latex equations are not rendered correctly and there is no live reload while editing.
- This package solves the above problems and you can launch it directly through the shortcuts:
  - Toggle Preview: `ctrl-shift-m`
  - Toggle Math Rendering: `ctrl-shift-x`
- You can also install and enable [Pandoc](https://pandoc.org/) with this package (it's optional).
- This package is really handy if you want to write technicalblog posts with math equations.

For more information and custom settings, visit the [atom.io page](https://atom.io/packages/markdown-preview-plus) and the [github repository](https://github.com/atom-community/markdown-preview-plus/) for this package.
