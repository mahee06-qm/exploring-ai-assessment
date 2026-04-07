import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_preds = pd.read_csv('final_automation_predictions.csv')
df_ab = pd.read_csv('AB.csv')

# Filter for 'Originality' Level
df_orig = df_ab[(df_ab['Element Name'] == 'Originality') & (df_ab['Scale Name'] == 'Level')]

# Merge
df_plot = pd.merge(df_preds[['O*NET-SOC Code', 'Resilience_Score']], 
                   df_orig[['O*NET-SOC Code', 'Data Value']], 
                   on='O*NET-SOC Code')

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot, x='Data Value', y='Resilience_Score', alpha=0.6)
plt.title('Relationship: Originality Level vs. Resilience Score')
plt.xlabel('Originality (Level)')
plt.ylabel('Resilience Score')
plt.show()