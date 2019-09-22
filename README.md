# Clustering-Headlines
Clustering and Matching News Headlines using K-means

A 75K corpus of news headlines scraped from a news channel's API was used to create a model that clusters these headlines and matches new 
headlines with the closest matches from the corpus.

K-means algorithm was used to cluster and determine the optimum number of clusters. Matching was done by finding the headline with the 
least euclidean distance from the processed search headline's coordinate point in the same cluster.

The headlines were cleaned and processed by lemmatizing, removing stop words and tokenizing. Features from these headlines
were subsequently vectorized and clustered using k-means elbow method to find the optimum number of clusters which was determined to be 3.

The new headline would be processed and fitted on the model to predict its cluster and best matches.
