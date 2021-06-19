# Import Pandas
import pandas as pd
import numpy as np

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Import linear_kernel (faster than cosine_similarities)
from sklearn.metrics.pairwise import linear_kernel


#fetch and return the data set
def get_data():
        df = pd.read_csv('./netflix_titles.csv' , encoding='UTF-8')
        return df

#combine chosen features into a single string
def merge_columns(movie):
    return movie['description']+" "+movie['type']+" "+movie['cast']+" "+movie['director']+movie['country']+movie['rating']+movie['listed_in']


#Select the needed features and combine them in one column:
def combine_features(df):
    features = ['description','type','director','cast','country','rating','listed_in']
    for feature in features:
        df[feature] = df[feature].fillna('')
    return df.apply(merge_columns,axis=1)

#Vectorize the data using TfidfVectorizer and calculate the cosine similarity between data row by column
def vectorize_data(combined_df):
    vector = TfidfVectorizer(   max_df=0.4,        
                            min_df=1,     
                            stop_words='english',
                            lowercase=True, 
                            use_idf=True,   
                            norm=u'l2',    
                            smooth_idf=True 
                            )
    return vector.fit_transform(combined_df)

#calculate the scalar products of multiple vectors
def scalar_product(vectors):
  product=vectors[0]
  i=1
  while i<vectors.get_shape()[0]:
    product=np.dot(product, vectors[i])
    i+=1
  return product


def recommendations(titles):
    
    df=get_data()
    df_combine=combine_features(df)
    tfidf_matrix=vectorize_data(df_combine)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if (len(titles)==0):
         print("Please select your favorite movie")
         return

    elif (len(titles)==1):
        movie_index = indices[titles[0]]
        # Compute the cosine similarity matrix
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)   
        #list of similar movies in the form of(movie_index, similarity_rate)
        similar_movies = list(enumerate(cosine_similarities[movie_index]))
        #sorting similar movies in descending order with eliminating the first element 
        #wich is the movie itself
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:6]
     
 
    else :
        product_vect=tfidf_matrix[indices[titles[0]]]
        indices_slected_titles=[]

        for e in titles:
            indices_slected_titles.append(indices[e])
            matrix_result = (product_vect.tocsr() +tfidf_matrix[indices[e]].tocsr()).tolil()
        modif_tfidf_matrix=tfidf_matrix
        modif_tfidf_matrix[0]= scalar_product(matrix_result )
        cosine_similarities = linear_kernel(modif_tfidf_matrix, tfidf_matrix)
        similar_movies = list(enumerate(cosine_similarities[0]))
        #sorting similar movies in descending order with eliminating the indexes of chosen movies
        check =True
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[:8]
        list_repeated_item=[]
        for e in sorted_similar_movies:
            for item in indices_slected_titles:
                if (e[0]==item):
                    check=False
                    list_repeated_item.append(e)
       
        if check==False:
            if len( list_repeated_item)==1:
                sorted_similar_movies.remove(list_repeated_item[0])
            if len( list_repeated_item)==2:
                for e in list_repeated_item:
                    sorted_similar_movies.remove(e)
         
    
    # Get the movie indices
    movie_indices = [i[0] for i in sorted_similar_movies[:5]]
    
    return df['title'].iloc[movie_indices].to_dict()
