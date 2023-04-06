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

5. Install the SNCosmo package in development mode. From the git directory::

      pip install -e .

   If you are only editing Python code, the latest code will be used when you
   import sncosmo in a Python interpreter for the first time. If you are
   editing any of the Cython code in SNCosmo (files with .c or .pyx
   extensions), then you will need to run this command again to compile that
   code for your changes to be picked up.


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


What happens when the upstream branch is updated?
-------------------------------------------------

Suppose that you are following the above workflow: you created a topic
branch ``simulation-enhancements`` and made a few commits on that
branch. You now want to create a pull request, but there's a problem:
while you were working, more commmits were added to the
``upstream/master`` branch on GitHub. The history of your branch has
now diverged from the main development branch! What to do?

1. Fetch the changes made to the upstream branch on so that you can
   deal with the changes locally::

       git fetch upstream

   This will update your local branch ``upstream/master`` (and any
   other ``upstream`` branches) to the match the state of the upstream
   branch on GitHub. It doesn't do any merging or resolving, it just
   makes the new changes to ``upstream/master`` visible locally.

2. There are two options for this next step: ``merge`` or ``rebase``
   with the latter being preferred for this purpose. Assuming you are
   on your branch ``simulation-enhancements``, you *could* do ``git
   merge upstream/master``. This would create a merge commit that
   merges the diverged histories back together. This works, but it can
   end up creating a confusing commit history, particularly if you
   repeat this process several times while working on your new
   branch. Instead, you can do::

       git rebase upstream/master

   This actually *rewrites* your commits to make it look like they
   started from where ``upstream/master`` now is, rather than where it
   was when you started work on your ``simulation-enhancements``
   branch. Your branch will have the exact same contents as if you had
   used ``git merge``, but the history will be different than it would
   have been if you had merged. In particular, there is no merge
   commit created, because the history has been rewritten so that your
   branch starts where ``upstream/master`` ends, and there is no
   divergent history to resolve.  This means you can rebase again and
   again without creating a convoluted history full of merges back and
   forth between the branches.


Trying out new ideas
--------------------

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

Testing
=======

SNCosmo uses pytest to check that all of the code is running as expected. When
you add new functionality to SNCosmo, you should write a test for that
functionality. All of the tests can be found in the ``sncosmo/tests``
directory.

When a new PR is created, the testsuite will be run automatically on a range
of different machines and conditions using `tox`. You can run these same tests
locally using ``tox``. First, install `tox`::

      pip install tox

From within the SNCosmo directory, run the test suite::

      tox -e py3

The previous command will run the core test suite with the currently installed
version of Python. You can run the full test suite with all of the optional
dependencies by adding the ``-alldeps`` tag::

      tox -e py3-alldeps

Running the tests with the ``-cov`` tag will generate a coverage report::

      tox -e py3-cov

SNCosmo includes hundreds of builtin bandpasses and sources that are downloaded
from external sites when they are loaded. ``tox`` can be used to check that all
of these builtins are accessible with the following command::

      tox -e builtins

``tox`` can also be used to check the code style::

      tox -e codestyle

or to build the documentation::

      tox -e build_docs

``tox`` uses virtual environments for testing which can be somewhat slow. You
can alternatively run the test in your own Python environment. First, install
all of the testing dependencies from the ``test`` section of ``setup.cfg``.
This can be done automatically when installing SNCosmo with the following
command::

      pip install -e .[test]

The tests can then be run with the following command::

      pytest --pyargs sncosmo


Developer's documentation: release procedure
============================================

The release procedure is automated through GitHub Actions. To create a new
release:

- Update ``docs/history.rst`` with a summary of the new version's changes.
- Bump version in ``sncosmo/__init__.py``.
- Ensure that the tests have all completed successfully and that the docs are
  looking good.
- Create a new release through the releases tab on GitHub, and tag it with the
  latest version.
- Copy the change list into the release description.
- Publish the release.

**Packaging and Docs**

- GitHub Actions will trigger after each release and build compiled wheels and
  source distributions. These will then be pushed to PyPI.
- A conda build should start (with some delay) via a bot pull request
  at https://github.com/conda-forge/sncosmo-feedstock. Merge the PR
  once it passes all tests.
- The docs for the release will show up on readthedocs.org as the new
  ``stable`` version.

**Bumping Minimum Supported Python Version**

Versions are hardcoded in 

- tox.ini - update the ``envlist = py{...}`` line.
- setup.cfg - update ``python_requires``, ``install_requires``, and ``oldestdeps`` as needed. 
- .github/workflows/run_tests.yml - update the ``python`` and 
  ``toxenv`` lines
- docs/install.rst - Ensure that the first line "SNCosmo works on 
  Python 3.x+" is correct