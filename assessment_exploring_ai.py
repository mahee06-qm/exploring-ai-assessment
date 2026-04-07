import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df_preds = pd.read_csv('final_automation_predictions.csv')
df_occ = pd.read_csv('Occupation_Data.csv')

# Merge
df_plot = pd.merge(df_preds[['O*NET-SOC Code', 'Resilience_Score']], 
                   df_occ[['O*NET-SOC Code', 'Title']], 
                   on='O*NET-SOC Code')

# Get Top 20
top_20 = df_plot.sort_values(by='Resilience_Score', ascending=False).head(20)

# Plot
plt.figure(figsize=(12, 8))
plt.barh(top_20['Title'], top_20['Resilience_Score'], color='teal')
plt.gca().invert_yaxis()
plt.title('Top 20 Most Resilient Occupations')
plt.xlabel('Resilience Score')
plt.show()