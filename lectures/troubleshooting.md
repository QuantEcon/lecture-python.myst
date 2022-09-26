---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(troubleshooting)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Troubleshooting

```{contents} Contents
:depth: 2
```

This page is for readers experiencing errors when running the code from the lectures.

## Fixing Your Local Environment

The basic assumption of the lectures is that code in a lecture should execute whenever

1. it is executed in a Jupyter notebook and
1. the notebook is running on a machine with the latest version of Anaconda Python.

You have installed Anaconda, haven't you, following the instructions in [this lecture](https://python-programming.quantecon.org/getting_started.html)?

Assuming that you have, the most common source of problems for our readers is that their Anaconda distribution is not up to date.

[Here's a useful article](https://www.anaconda.com/blog/keeping-anaconda-date)
on how to update Anaconda.

Another option is to simply remove Anaconda and reinstall.

You also need to keep the external code libraries, such as [QuantEcon.py](https://quantecon.org/quantecon-py) up to date.

For this task you can either

* use conda install -y quantecon on the command line, or
* execute !conda install -y quantecon within a Jupyter notebook.

If your local environment is still not working you can do two things.

First, you can use a remote machine instead, by clicking on the Launch Notebook icon available for each lecture

```{image} _static/lecture_specific/troubleshooting/launch.png

```

Second, you can report an issue, so we can try to fix your local set up.

We like getting feedback on the lectures so please don't hesitate to get in
touch.

## Reporting an Issue

One way to give feedback is to raise an issue through our [issue tracker](https://github.com/QuantEcon/lecture-python/issues).

Please be as specific as possible.  Tell us where the problem is and as much
detail about your local set up as you can provide.

Another feedback option is to use our [discourse forum](https://discourse.quantecon.org/).

Finally, you can provide direct feedback to [contact@quantecon.org](mailto:contact@quantecon.org)

