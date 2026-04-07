import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_preds = pd.read_csv('final_automation_predictions.csv')
df_ab = pd.read_csv('AB.csv')

# Filter for 'Manual Dexterity' Level
df_dex = df_ab[(df_ab['Element Name'] == 'Manual Dexterity') & (df_ab['Scale Name'] == 'Level')]

# Merge
df_plot = pd.merge(df_preds[['O*NET-SOC Code', 'Resilience_Score']], 
                   df_dex[['O*NET-SOC Code', 'Data Value']], 
                   on='O*NET-SOC Code')

# Plot
plt.figure(figsize=(10, 6))
sns.regplot(data=df_plot, x='Data Value', y='Resilience_Score', 
            scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
plt.title('Manual Dexterity Level vs. Resilience Score')
plt.xlabel('Manual Dexterity (Level)')
plt.ylabel('Resilience Score')
plt.show()