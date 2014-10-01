************
Contributing
************

Overview
========

SNCosmo follows the same general development workflow as astropy and
many other open-source software projects. The `astropy development
workflow`_ page documents the process in some detail.  While you
should indeed read that page, it can seem a bit overwhelming at first.
So, we present here a rough outline of the process, and try to explain
the reasoning behind it.

.. _`astropy development workflow`: http://astropy.readthedocs.org/en/v0.4.1/development/workflow/development_workflow.html

The process is centered around git and GitHub, so you need to know how
to use basic git commands and also have a GitHub account. There is a
"blessed" copy of the repository at
https://github.com/sncosmo/sncosmo. Individual contributors make
changes to their own copy (or "fork" or "clone" in git parlance) of
the repository, e.g., https://github.com/kbarbary/sncosmo, then ask
that their changes be merged into the "blessed" copy via a Pull
Request (PR) on GitHub. A maintainer (currently Kyle) will review
the changes in the PR, possibly ask for alterations, and then
eventually merge the change.

This seems overly complex at first glance, but there are two main
benefits to this process: (1) Anyone is free to try out any crazy
change they want and share it with the world on their own GitHub account,
without affecting the "blessed" repository, and (2) Any proposed
changes are reviewed and discussed by at least one person (the
maintainer) before being merged in.

Detailed steps
==============

Do once:
--------

1. Hit the "fork" button in the upper right hand corner of the
   https://github.com/sncosmo/sncosmo page. This creates a clone of the
   repository on your personal github account.

2. Get it onto your computer (replace username with your GitHub username)::

       git clone git@github.com:username/sncosmo.git

3. Add the "blessed" version as a remote::

       git remote add upstream git@github.com:sncosmo/sncosmo.git

   This will allow you to update your version to reflect new changes to
   the blessed repository that others have made).

4. Check that everything is OK::

       $ git remote -v
       origin    git@github.com:username/sncosmo.git (fetch)
       origin    git@github.com:username/sncosmo.git (push)
       upstream    git@github.com:sncosmo/sncosmo.git (fetch)
       upstream    git@github.com:sncosmo/sncosmo.git (push)

   You can call the remotes anything you want. "origin" and "upstream"
   have no intrinsic meaning for git; they're just nicknames. The
   astropy documentation calls them "your-github-username" and
   "astropy" respectively.


Every time you want to make a contribution:
-------------------------------------------

1. Ensure that the clone of the repository on your local machine is
   up-to-date with the latest upstream changes by doing ``git fetch
   upstream``. This updates your local "remote tracking branch", called
   ``upstream/master``.

2. Create a "topic branch" for the change you want to make. If you plan
   to make enhancements to the simulation code, name the branch
   something like "simulation-enhancements"::
 
       git branch simulation-enhancements upstream/master

   (``upstream/master`` is where the branch branches off from.)

3. Move to the branch you just created::

       git checkout simulation-enhancements

4. *Make changes, ensure that they work, etc. Make commits as you go.* 

5. Once you're happy with the state of your branch, push it to your
   GitHub account for the world to see::

       git push origin simulation-enhancements

6. Create a PR: Go to your copy on github
   (https://github.com/username/sncosmo) select the branch you just
   pushed in the upper left-ish of the page, and hit the green button
   next to it. (It has a mouse-over "compare, review, create a pull
   request")


Trying out new ideas
====================

git branches are the best way to try out new ideas for code
modifications or additions. You don't even have to tell anyone about
your bad ideas, since branches are local!  They only become world
visible when you push them to GitHub. If, after making several
commits, you decide that your new branch ``simulation-enhancements``
sucks, you can just create a new branch starting from upstream/master
again. If it is a really terrible idea you never want to see again,
you can delete it by doing ``git branch -D simulation-enhancements``.

Obviously this isn't a complete guide to git, but hopefully it
jump-starts the git learning process.
