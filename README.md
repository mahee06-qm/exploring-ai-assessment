# O*NET Occupational Automation Predictor

## Project Overview
This project explores the intersection of machine learning and the future of work. I developed a **Support Vector Machine (SVM)** classifier to predict whether 894 different occupations are "Resilient" (safe) or "Augmentable" (at risk) based on their specific skill requirements. By analyzing 93 high-dimensional variables from the O*NET database, the model identifies the "human bottlenecks" of automation—like creativity and social intelligence—that shield certain careers from displacement.

## Technical Workflow
To ensure my results were reliable and scientifically honest, I moved beyond simple scripting to a professional data science pipeline:

1.  **Data Cleaning:** I merged occupational datasets and filtered for "Importance" scales. This resolved a "Dimensionality Conflict" where redundant data categories were causing initial code failures.
2.  **The Pipeline:** I integrated a `StandardScaler` and `SVC` into a formal `Pipeline`. This was a critical iterative improvement that prevented "data leakage," ensuring the model didn't "cheat" by seeing test data during the training phase.
3.  **Algorithmic Choice:** I chose a **Radial Basis Function (RBF) Kernel**. While more complex than a linear model, it was essential for capturing the non-linear, "pockets of safety" in job tasks that simple models miss.
4.  **Optimization:** I used `GridSearchCV` to mathematically find the best settings ($C=10, \gamma=0.01$) rather than just guessing, which significantly boosted the model's reliability.



## Key Results
* **Accuracy:** The model achieved a **91% overall accuracy**.
* **Discrimination Power:** A **ROC-AUC score of 0.96** confirms the model has an excellent ability to distinguish between safe and risky jobs.
* **Human Insights:** Using `permutation_importance`, I found that skills involving complex perception and social negotiation are the strongest predictors of career resilience.



## Responsible Use of AI
I utilized Generative AI (Gemini and ChatGPT) as technical mentors during this project. Specifically:
* **Gemini** helped troubleshoot the complex Python syntax for data pivoting and the custom data dictionary.
* **ChatGPT** assisted in structuring the initial Pipeline code to ensure proper normalization.

I manually validated every line of AI-generated code to ensure it met university standards for technical rigor. While AI was useful for syntax, I remained the "architect," ensuring the model accounted for the ethical responsibility of predicting human career paths.

## Limitations
It is important to note that this model uses static O*NET data. It cannot account for sudden, future technological breakthroughs that might change which skills are "bottlenecks" in the years to come.

## How to Use
1.  Ensure you have `pandas`, `sklearn`, and `numpy` installed.
2.  Run the `Exploring_AI_allcodes.ipynb` notebook.
3.  The final output will display the **Optimal Parameters** and a **Classification Report**.
