# AI Code Exploration

Just a place to dump some of the stuff I've been working on for my AI module. It's mostly just testing out some predictive modeling for "Resilience" scores using basic machine learning libraries.

## What's in here?

The notebook covers a few main areas:
* **Data Prep**: Loading up the dataset and getting it ready for the model.
* **Grid Search**: Using `GridSearchCV` to find the best settings for the model. I used 'f1' as the scoring metric because the classes seemed a bit unbalanced.
* **Evaluation**: Checking how the model actually performed. It prints out a classification report and the ROC-AUC score (which came out to about 0.81 in my last run).
* **Interpretability**: I used permutation importance to see which features actually matter.

## Key Parameters

I've been focusing on two main parameters for the risk assessment part of the code:
1.  **Probability**: The likelihood of a specific event occurring.
2.  **Risk_Level**: The classified level of risk associated with the predictive outcome.

## How to use it

1.  Make sure you have `sklearn`, `numpy`, and `pandas` installed.
2.  Open `Exploring_AI_codes.ipynb` in Jupyter.
3.  Run the cells. The last cell is the most important one—it tells you the "Top Predictive Skill for Resilience" based on the model's importance scoring.
