import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import collections

def GetDataHistory():
	Construct_CSV # it contains various rows containg VideoID and text (tags, category, title) from user's history
	return CSVTable

def GetData():
	Construct_CSV # it contains various rows containg VideoID and text (tags, category, title) randomly sampled from database
	return CSVTable

def clean_text(raw_text):
    stemming = PorterStemmer()
	stops = set(stopwords.words("english"))

    # Convert to lower case
    text = raw_text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in stemmed_words if not w in stops]
    
    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words

def get_vectors(data):
	data['feature_attached'] = df['tags'].astype(str)+' '+df['category']+' '+df['title']
	data['feature_combined'] = clean_text(data['features_attached'])

	data.drop(df.columns['feature_attached']) #delete the unprocessed column 

	vectorizer = TfidfVectorizer()
	TFidf = vectorizer.fit_transform(data['feature_combined'])
	col_of_tf = vectorizer.get_feature_names()
	#print(TFidf.toarray())


	model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  

	new = dict()# this dictionary will contain VideoID as key and value will be vectorized form of text (tags, category, title)
	for index, row in data.iterrows():
		vector=[]

		for feature in row['feature_combined']:
			vector+=model[feature]*TFidf.toarray()[index][col_of_tf.index(feature)]#add all the vectors of words for a video

		vector=vector/len(row['feature_combined']) #take the average of every element of the vector
		new[row['VideoID']] = vector

	return new

def cosine_similarity(vec1,vec2):#return cos similarity 
    sum11, sum12, sum22 = 0, 0, 0
    for i in range(len(vec1)):
        x = vec1[i]; y = vec2[i]
        sum11 += x*x
        sum22 += y*y
        sum12 += x*y
    return sum12/math.sqrt(sum11*sum22)


data = GetDataHistory()#videos from user's history
data_vec_dict = get_vectors(data)#vectorized form

data_new = GetData()#videos randomly sampled from database
data_new_vec_dict = get_vectors(data_new)#vectorized form

res = {}
for key,value in data_vec_dict.items():
    temp,tempDict= 0,{}
    for keyC,valueC in data_new_vec_dict.items():
        if keyC == key:#VideoID same, therefore same video (previously watched)
            continue
        temp = cosine_similarity(value,valueC)
        tempDict[keyC] = temp
    res[key]= tempDict

n = 50 #number of recommendations we want to display to the user
collections.Counter(res).most_common(n) # returns the 'n' highest values from the dictionary

final_result = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)} #decending sort videos while preserving key (VideoID)
print(final_result)#top n recommended to the user









