#Importing dependencies
import numpy as np
import pandas as pd #import pandas
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from IPython.display import display

customer_data = pd.read_csv('RiversideMall-Customers.csv')
#show dataset
display(customer_data.head())

#rows and cols
print(customer_data.shape)

#info about frames
customer_data.info()

customer_data.isnull().sum()

X= customer_data.iloc[: ,[3,4]].values
print(X)

#choosing number of clusters
#WCSS

WCSS =[]

for i in range(1, 11):
    means =KMeans(n_clusters=i, init='k-means++', random_state=42)
    means.fit(X)
    
    WCSS.append(means.inertia_)

#plot an elbow graph

sns.set()
plt.plot(range(1,11), WCSS)
plt.title('The elbow point graph')
plt.xlabel('# of clusters')
plt.ylabel('within cluster sum squares')
#plt.show()

#the optimum number of clusters is the one where there is no significant drop

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

#return a  label for each data point based of their cluster

Y = kmeans.fit_predict(X)

print(Y)

#Visualization of all the clusters..

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='blue', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='yellow', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='violet', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label ='Centroids')

plt.title('Customer Visualization')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
