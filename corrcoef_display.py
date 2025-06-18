import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Step 1: Read the JSONL file
jsonl_file = 'correlation_coefficients_save.jsonl'  # Path to your JSONL file
data = []

# Load JSONL file
with open(jsonl_file, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Pivot DataFrame to create a similarity matrix
similarity_matrix = df.pivot(index='item1', columns='item2', values='similarity')

# Step 3: Create a labeled heatmap using seaborn
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Step 4: Customize labels
plt.title('Similarity Heatmap', fontsize=16)
plt.xlabel('Item 2', fontsize=12)
plt.ylabel('Item 1', fontsize=12)

# Show the heatmap
plt.show()