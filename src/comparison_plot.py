import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into a DataFrame
df = pd.read_csv(r'C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\outputs\output_time.csv')

# Plot Execution Time
plt.figure(figsize=(8, 6))
plt.bar(df['Algorithm'], df['Execution Time'], color=['blue', 'green'])
plt.title('Execution Time Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (s)')
plt.show()

# Plot Number of Clusters
plt.figure(figsize=(8, 6))
plt.bar(df['Algorithm'], df['Number of Clusters'], color=['blue', 'green'])
plt.title('Number of Clusters Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Number of Clusters')
plt.show()


df = pd.read_csv(r'C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\outputs\output_louvain_imps.csv')
plt.figure(figsize=(8, 6))
plt.bar(df['Algorithm'], df['Execution Time'], color=['blue', 'green'])
plt.title('Execution Time Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (s)')
plt.show()

# Plot Number of Clusters
plt.figure(figsize=(8, 6))
plt.bar(df['Algorithm'], df['Number of Clusters'], color=['blue', 'green'])
plt.title('Number of Clusters Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Number of Clusters')
plt.show()
