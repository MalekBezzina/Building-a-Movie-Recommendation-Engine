import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
### helper functions use  when needed :p ###

def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]
####################################################################################
    
##step 1 :import dataset 
df =pd.read_csv("movie_dataset.csv")
#print(df)

##step 2 : select features

features=['keywords','cast','genres','director']

###step 3 : create column to combine features

## getting rid of all the NAN values 
for feature in features:
    df[feature]=df[feature].fillna('')

def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
df["combine_features"]=df.apply(combine_features,axis=1)

#print(df["combine_features"].head())

##step 4 : create count matrix 

cv=CountVectorizer()

count_matrix=cv.fit_transform(df["combine_features"])
#print (count_matrix.toarray())

###step 5 : cosine similarity

cosine_sim = cosine_similarity(count_matrix)
#print (cosine_sim)

movie_user_likes="Sheena" #the choice u made before the one we're building on 

###step 6 : get index of movie

movie_index=get_index_from_title(movie_user_likes)

similar_movies=list(enumerate(cosine_sim[movie_index]))

#print(similar_movies)

###step 7 : get a list of similar movies

sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

#rint(sorted_similar_movies)
### step 8 : print titles of first 50 movies

i=0
for element in sorted_similar_movies:
		print (get_title_from_index(element[0]))
		i=i+1
		if i>50:
			break













