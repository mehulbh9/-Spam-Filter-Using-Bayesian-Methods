# Spam Filter Using Bayesian Methods

This project contains a Python script that implements a spam filter using Bayesian probability methods. The script processes email data to classify messages as either spam or ham (non-spam) based on their content.

## Project Description

The script `spam_filter.py` uses a Naive Bayes classifier to learn the distribution of words in spam and ham emails and then classifies new emails based on these learned distributions. It calculates the likelihood of an email being spam or ham and uses the Maximum A Posteriori (MAP) rule to determine the final classification.

## Features

- Learns word distributions from training data.
- Classifies new emails as spam or ham.
- Calculates posterior probabilities for the classifications.
- Measures performance based on a test dataset.
