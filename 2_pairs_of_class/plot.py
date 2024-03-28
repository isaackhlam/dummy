import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

mean_scores_matrix = np.zeros((10, 10))
top_k_scores_matrix = np.zeros((10, 10))
precision_matrix = np.zeros((10, 10))
recall_matrix = np.zeros((10, 10))
density_matrix = np.zeros((10, 10))
coverage_matrix = np.zeros((10, 10))

for filename in os.listdir('./'):
    # Extract the class numbers
    if filename.endswith("score.txt"):
        _, gen_class, real_class, _ = filename.split('_')
        _, real_class = real_class.split('against')
        # real_class = real_class.removesuffix(".txt")
        real_class, gen_class = int(real_class), int(gen_class)
        # Get the mean score 
        score = np.loadtxt(filename)
        mean_score = np.mean(score[score!=0])
        # Get top k (30%) score
        sorted_score = sorted(score, reverse=True)
        top_k_score = np.mean(sorted_score[:int(0.3 * len(sorted_score))])

        top_k_scores_matrix[real_class][gen_class] = top_k_score
        mean_scores_matrix[real_class][gen_class] = mean_score
    
    if filename.endswith("prdc.json"):
        _, gen_class, real_class, _ = filename.split('_')
        _, real_class = real_class.split('against')
        real_class, gen_class = int(real_class), int(gen_class)
        with open(filename, "r") as f:
            prdc = json.load(f)
        f.close()
        
        precision_matrix[real_class][gen_class] = prdc["precision"]
        recall_matrix[real_class][gen_class] = prdc["recall"]
        density_matrix[real_class][gen_class] = prdc["density"]
        coverage_matrix[real_class][gen_class] = prdc["coverage"]

# Plot the heatmap of mean score
plt.figure(figsize=(10,8))
sns.heatmap(mean_scores_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=classDict.keys(), yticklabels=classDict.keys())
plt.title('Mean Rarity Scores without 0 scores')
plt.xlabel('Generated Image Class')
plt.ylabel('Real Manifold Class')
plt.show()

# Plot the heatmap of top k score
plt.figure(figsize=(10,8))
sns.heatmap(top_k_scores_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=classDict.keys(), yticklabels=classDict.keys())
plt.title('Top 30% Rarity Scores')
plt.xlabel('Generated Image Class')
plt.ylabel('Real Manifold Class')
plt.show()

# Plot the heatmap of precision
plt.figure(figsize=(10,8))
sns.heatmap(precision_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=classDict.keys(), yticklabels=classDict.keys())
plt.title('Precision')
plt.xlabel('Generated Image Class')
plt.ylabel('Real Manifold Class')
plt.show()

# Plot the heatmap of recall
plt.figure(figsize=(10,8))
sns.heatmap(recall_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=classDict.keys(), yticklabels=classDict.keys())
plt.title('Recall')
plt.xlabel('Generated Image Class')
plt.ylabel('Real Manifold Class')
plt.show()

# Plot the heatmap of density
plt.figure(figsize=(10,8))
sns.heatmap(density_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=classDict.keys(), yticklabels=classDict.keys())
plt.title('Density')
plt.xlabel('Generated Image Class')
plt.ylabel('Real Manifold Class')
plt.show()

# Plot the heatmap of coverage
plt.figure(figsize=(10,8))
sns.heatmap(coverage_matrix, annot=True, fmt=".2f", cmap='viridis',
            xticklabels=classDict.keys(), yticklabels=classDict.keys())
plt.title('Coverage')
plt.xlabel('Generated Image Class')
plt.ylabel('Real Manifold Class')
plt.show()

print(np.mean(mean_scores_matrix))
print(mean_scores_matrix)