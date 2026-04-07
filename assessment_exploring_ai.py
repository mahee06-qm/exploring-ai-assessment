import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the prediction data
df = pd.read_csv('final_automation_predictions.csv')

# --- POINT 4: Mean Resilience Score by SOC Occupational Group ---

# 1. Extract SOC Major Group (first 2 digits of the code)
df['SOC_Major_Group'] = df['O*NET-SOC Code'].str[:2]

# 2. Define SOC Group Mapping
soc_mapping = {
    '11': 'Management', '13': 'Business/Financial', '15': 'Computer/Math',
    '17': 'Arch/Eng', '19': 'Science', '21': 'Social Service',
    '23': 'Legal', '25': 'Education', '27': 'Arts/Media',
    '29': 'Healthcare Tech', '31': 'Healthcare Support', '33': 'Protective Service',
    '35': 'Food Prep', '37': 'Cleaning/Maint', '39': 'Personal Care',
    '41': 'Sales', '43': 'Admin Support', '45': 'Farming/Forestry',
    '47': 'Construction', '49': 'Maintenance/Repair', '51': 'Production',
    '53': 'Transportation'
}
df['Group_Name'] = df['SOC_Major_Group'].map(soc_mapping)

# 3. Calculate Mean Resilience Score per Group
group_resilience = df.groupby('Group_Name')['Resilience_Score'].mean().sort_values()

# 4. Plot
plt.figure(figsize=(12, 8))
group_resilience.plot(kind='barh', color='skyblue')
plt.title('Mean Resilience Score by Occupational Group')
plt.xlabel('Average Resilience Score (Higher = More Resilient)')
plt.ylabel('Occupational Group')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- POINT 5: Radar Chart: Skill Profile Comparison ---

# 1. Select Key Representative Features (based on importance)
radar_features = [
    'Originality', 
    'Thinking Creatively', 
    'Assisting and Caring for Others', 
    'Analyzing Data or Information', 
    'Making Decisions and Solving Problems',
    'Manual Dexterity'
]

# 2. Calculate means for Resilient vs Augmentable
comparison = df.groupby('Predicted_Risk_Level')[radar_features].mean().reset_index()

# 3. Prepare data for Radar Chart
labels = np.array(radar_features)
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Close the circle

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for index, row in comparison.iterrows():
    values = row[radar_features].values.flatten().tolist()
    values += values[:1] # Close the circle
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Predicted_Risk_Level'])
    ax.fill(angles, values, alpha=0.25)

# Fix axis labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

plt.title('Skill Profile: Resilient vs. Augmentable Jobs', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()