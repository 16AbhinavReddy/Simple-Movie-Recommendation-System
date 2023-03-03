import gradio as gd
import pandas as pd
import numpy as np
# import seaborn as sbn
# import matplotlib.pyplot as plt
def function(movie):
    data = pd.read_csv("movie_metadata.csv")
    # data.columns
    data = data.fillna(" ")
    data["index"] = data.index
    def get_title_from_index(index):
        return data[data.index == index]["movie_title"].values[0]

    def get_index_from_title(title):
        return data[data.movie_title == title]["index"].values[0]

    features = ['director_name','actor_1_name','plot_keywords','genres','language', 'country','imdb_score']
    def combining_features(data):
        return str(data['director_name']) + " " + str(data['actor_1_name']) + " " + str(data['plot_keywords']) + " " + str(data['genres']) + " " + str(data['language']) + " " + str(data['country']) + str(data['imdb_score'])

    data["Combined Features"] = data.apply(combining_features, axis= 1)
    # data.head()



    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    model = CountVectorizer()
    count_matrix = model.fit_transform(data["Combined Features"])
    cosine_sim = cosine_similarity(count_matrix)

    index1 = get_index_from_title(str(movie))

    similar_movies =list(enumerate(cosine_sim[index1]))

    sorted_by_sim = sorted(similar_movies, key=lambda x:x[1], reverse= True)

    i = 0
    movie_list = "  "
    for movie in sorted_by_sim:
        # print(get_title_from_index(movie[0]))
        movie_list = movie_list + get_title_from_index(movie[0]) + "\n"
        i+=1
        if i>20:
            break
    return movie_list
def run(movie):
    movie = movie+"\xa0"
    final = function(movie)
    return final
outputs = gd.outputs.Textbox()
app = gd.Interface(fn=run, inputs="text", outputs=outputs,description="This is a movie recommendation model")
app.launch(share=True)
    
