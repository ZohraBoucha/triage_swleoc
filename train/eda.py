import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the CSV file
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset.csv', encoding='latin-1')

# Filter the DataFrame to include only 'surgery' and 'discharge' outputs
df = df[df['output'].isin(['surgery', 'discharge'])]

# Function to balance dataset
def balance_dataset(dataset):
    surgery = dataset[dataset['output'] == 'surgery']
    discharge = dataset[dataset['output'] == 'discharge']
    min_count = min(len(surgery), len(discharge))
    balanced_surgery = surgery.sample(min_count, random_state=42)
    balanced_discharge = discharge.sample(min_count, random_state=42)
    return pd.concat([balanced_surgery, balanced_discharge]).sample(frac=1, random_state=42)

# Create balanced original dataset
balanced_original_df = balance_dataset(df)

# Perform stratified split on unbalanced data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['output'], random_state=42)

# Create balanced datasets for train and test
balanced_train_df = balance_dataset(train_df)
balanced_test_df = balance_dataset(test_df)

# Save to CSV files
df.to_csv('triage_original_unbalanced.csv', index=False)
balanced_original_df.to_csv('triage_original_balanced.csv', index=False)
train_df.to_csv('triage_train_unbalanced.csv', index=False)
test_df.to_csv('triage_test_unbalanced.csv', index=False)
balanced_train_df.to_csv('triage_train_balanced.csv', index=False)
balanced_test_df.to_csv('triage_test_balanced.csv', index=False)

# Count samples for each set
original_surgery = len(df[df['output'] == 'surgery'])
original_discharge = len(df[df['output'] == 'discharge'])
bal_original_surgery = len(balanced_original_df[balanced_original_df['output'] == 'surgery'])
bal_original_discharge = len(balanced_original_df[balanced_original_df['output'] == 'discharge'])
train_surgery = len(train_df[train_df['output'] == 'surgery'])
train_discharge = len(train_df[train_df['output'] == 'discharge'])
test_surgery = len(test_df[test_df['output'] == 'surgery'])
test_discharge = len(test_df[test_df['output'] == 'discharge'])
bal_train_surgery = len(balanced_train_df[balanced_train_df['output'] == 'surgery'])
bal_train_discharge = len(balanced_train_df[balanced_train_df['output'] == 'discharge'])
bal_test_surgery = len(balanced_test_df[balanced_test_df['output'] == 'surgery'])
bal_test_discharge = len(balanced_test_df[balanced_test_df['output'] == 'discharge'])

# Plotting
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

# Original unbalanced dataset
ax1.bar(['Surgery', 'Discharge'], [original_surgery, original_discharge])
ax1.set_title('Original Unbalanced Dataset')
ax1.set_ylabel('Number of samples')

# Original balanced dataset
ax2.bar(['Surgery', 'Discharge'], [bal_original_surgery, bal_original_discharge])
ax2.set_title('Original Balanced Dataset')

# Unbalanced Train set
ax3.bar(['Surgery', 'Discharge'], [train_surgery, train_discharge])
ax3.set_title('Unbalanced Train Set')
ax3.set_ylabel('Number of samples')

# Balanced Train set
ax4.bar(['Surgery', 'Discharge'], [bal_train_surgery, bal_train_discharge])
ax4.set_title('Balanced Train Set')

# Unbalanced Test set
ax5.bar(['Surgery', 'Discharge'], [test_surgery, test_discharge])
ax5.set_title('Unbalanced Test Set')
ax5.set_ylabel('Number of samples')

# Balanced Test set
ax6.bar(['Surgery', 'Discharge'], [bal_test_surgery, bal_test_discharge])
ax6.set_title('Balanced Test Set')

for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
    for i, v in enumerate(ax.patches):
        ax.text(v.get_x() + v.get_width()/2, v.get_height(), str(int(v.get_height())),
                ha='center', va='bottom')
    ax.set_ylim(0, max(original_surgery, original_discharge) * 1.1)  # Set y-axis limit

plt.tight_layout()
plt.savefig('class_distribution_all_balanced.png')
plt.show()

print("\nPlot saved as 'class_distribution_all_balanced.png'")

# Print summaries
print("\nOriginal Unbalanced Dataset:")
print("Surgery samples:", original_surgery)
print("Discharge samples:", original_discharge)

print("\nOriginal Balanced Dataset:")
print("Surgery samples:", bal_original_surgery)
print("Discharge samples:", bal_original_discharge)

print("\nUnbalanced Train set:")
print("Surgery samples:", train_surgery)
print("Discharge samples:", train_discharge)

print("\nBalanced Train set:")
print("Surgery samples:", bal_train_surgery)
print("Discharge samples:", bal_train_discharge)

print("\nUnbalanced Test set:")
print("Surgery samples:", test_surgery)
print("Discharge samples:", test_discharge)

print("\nBalanced Test set:")
print("Surgery samples:", bal_test_surgery)
print("Discharge samples:", bal_test_discharge)
'''
Original dataset:
Total samples: 237
Surgery samples: 73
Discharge samples: 63

Original class distribution:
output
surgery      0.308017
discharge    0.265823
other        0.210970
tests        0.075949
fu           0.050633
physio       0.037975
referral     0.025316
injection    0.025316
Name: proportion, dtype: float64

--------------------------------------------------

After splitting:
Total samples: 136
Train samples: 108
Test samples: 28

Train set:
Surgery samples: 58
Discharge samples: 50

Test set:
Surgery samples: 15
Discharge samples: 13

Train class distribution:
output
surgery      0.537037
discharge    0.462963
Name: proportion, dtype: float64

Test class distribution:
output
surgery      0.535714
discharge    0.464286
'''