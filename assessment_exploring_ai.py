import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_preds = pd.read_csv('final_automation_predictions.csv')
df_wa = pd.read_csv('WA.csv')

# Filter for 'Assisting and Caring for Others' Level
df_care = df_wa[(df_wa['Element Name'] == 'Assisting and Caring for Others') & (df_wa['Scale Name'] == 'Level')]

# Merge
df_plot = pd.merge(df_preds[['O*NET-SOC Code', 'Predicted_Risk_Level']], 
                   df_care[['O*NET-SOC Code', 'Data Value']], 
                   on='O*NET-SOC Code')

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_plot, x='Predicted_Risk_Level', y='Data Value', palette='Set2')
plt.title('Assisting and Caring for Others by Risk Level')
plt.xlabel('Predicted Risk Level')
plt.ylabel('Assisting and Caring for Others (Level)')
plt.show()