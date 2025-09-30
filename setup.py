#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="rex_omni",
    version="1.0.0",
    author="IDEA Research",
    author_email="contact@idea.edu.cn",
    description="A high-level wrapper for Rex-Omni multimodal language model supporting various vision tasks",
    packages=find_packages(),
    python_requires=">=3.8",
)
