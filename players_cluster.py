import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore

df = pd.read_csv('./data/Seasons_Stats.csv')  
df = df[df['Year'].between(2012, 2017)]

keep_cols = ['Player', 'Pos', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV',
             'FG%', '3P%', 'FT%', 'G', 'MP']
df = df[keep_cols].dropna()
df = df.drop_duplicates(subset='Player')

features = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV', 'FG%', '3P%', 'FT%', 'G', 'MP']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

pca_weights = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
print("\nContribution of Stats to Principal Components (Higher = More Influence):")
print(pca_weights.T.round(3).sort_values(by='PC1', ascending=False))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', style='Pos', palette='Set2')
plt.title('NBA Player Clusters (2012–2017)\nColored by Cluster, Shaped by Position')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

crosstab = pd.crosstab(df['Pos'], df['Cluster'])
print("\nPosition vs. Cluster Distribution:")
print(crosstab)

crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)
crosstab_norm.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8, 5))
plt.title('Distribution of Clusters Within Each Position (2012–2017)')
plt.ylabel('Proportion of Players')
plt.xlabel('Position')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

unexpected_stats = {
    'C': ['AST', '3P%', 'STL'],
    'PF': ['3P%', 'AST'],
    'SF': ['BLK'],
    'SG': ['BLK', 'TRB'],
    'PG': ['BLK', 'TRB']
}

df_z = df.copy()
for stat in features:
    df_z[stat + '_z'] = df.groupby('Pos')[stat].transform(zscore)

unusual_players = []
for pos, stats in unexpected_stats.items():
    for stat in stats:
        mask = (df_z['Pos'] == pos) & (df_z[stat + '_z'] > 1.5)
        outliers = df_z[mask]
        for _, row in outliers.iterrows():
            unusual_players.append({
                'Player': row['Player'],
                'Position': pos,
                'Stat': stat,
                'Value': row[stat],
                'z-Score': row[stat + '_z'],
                'Cluster': row['Cluster']
            })

unusual_df = pd.DataFrame(unusual_players)
print("\nPlayers Who Excel in Unexpected Traits for Their Position:")
print(unusual_df.sort_values(by='z-Score', ascending=False).head(15))

unusual_df.to_csv('unexpected_role_players.csv', index=False)