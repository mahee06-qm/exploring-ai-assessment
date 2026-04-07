import pandas as pd

# 1. Load the Excel file
# Make sure the filename matches exactly what is in your Jupyter directory
df_titles = pd.read_csv('Occupation_Data.csv')

# 2. CLEAN THE COLUMNS (This is the fix!)
# This removes hidden spaces like "Title " or " O*NET-SOC Code"
df_titles.columns = df_titles.columns.str.strip()

# 3. Double check the names are now clean
print("Cleaned Columns:", df_titles.columns.tolist())

# 4. Create the dictionary
# Now 'O*NET-SOC Code' and 'Title' should work perfectly
soc_to_title = dict(zip(df_titles['O*NET-SOC Code'], df_titles['Title']))

# 5. Map the names to your automation results
df_results = pd.read_csv('final_automation_predictions.csv')
df_results['Job_Title'] = df_results['O*NET-SOC Code'].map(soc_to_title)

# 6. Show the results
print("\nSuccess! Here are the first 5 rows with Job Titles:")
display(df_results[['O*NET-SOC Code', 'Job_Title', 'Predicted_Risk_Level']].head(1100))