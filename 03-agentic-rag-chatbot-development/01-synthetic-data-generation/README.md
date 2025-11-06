# Synthetic Data Generation

This directory contains the implementation for generating and validating synthetic training data for the Canvas Assistant chatbot.

## Overview

The synthetic data generation process creates high-quality, diverse training data to improve the chatbot's ability to understand and respond to user queries about Canvas LMS.

## Components

### Data Generation Process
1. `00-generate-synthetic-data-basics-guide.ipynb`
   - Generates basic Canvas usage scenarios
   - Creates foundational question-answer pairs

2. `01-generate-synthetic-data-student-guides.ipynb`
   - Generates advanced student-specific scenarios
   - Creates complex, multi-step instruction pairs

3. `02-questions_quality_check.ipynb`
   - Validates generated questions for quality and relevance


## Output Datasets

Located in `../data/datasets/`:
- `synthetic_dataset-basics-guide.csv`
- `synthetic_dataset-student-guide.csv`
