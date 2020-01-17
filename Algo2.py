def GetData():
	Construct_CSV # it contains various rows of UserID, VideoID and Label
	return CSVTable

def GetVideos(user_id):
	Sample_Videos_User_has_Not_Seen_withIDEqualsUser_id
	return VideoID_List

def Recommend_Videos(model,user_id,n_recommended):
	VideoID_List = GetVideos(user_id)

	y_pred = np.round(model.predict([user_id, VideoID_List]),0) # get the score for unseen videos of this user
	arr = sorted(range(len(y_pred)), key=lambda k: y_pred[k], reverse=True) #return indices of items which are sorted according to higest values
	
	recommended_list = [] 
	for i in arr: 
		recommended_list.append(VideoID_List[i])

	return recommended_list[:n_recommended] # return the video ID of first "n_recommended" recommendations 

def Model_Recommendation(n_latent_factor_users,n_latent_factor_video,n_latent_factors_mf,Data):
	
	n_users = len( Data.UserID.Unique())	#get number of unique users
	n_videos = len( Data.VideosID.Unique())	#get number of unique videos

	video_input = keras.layers.Input(shape=[1],name='Video_Input')

	video_embedding_mlp = keras.layers.Embedding( input_dim = n_videos + 1, output_dim = n_latent_factor_video, name = "Video_Embedding_MLP")(video_input)
	video_vec_mlp = keras.layers.Flatten(name='FlattenVideos-MLP')(video_embedding_mlp)
	video_vec_mlp = keras.layers.Dropout(0.4)(video_vec_mlp)

	video_embedding_mf = keras.layers.Embedding( input_dim = n_videos + 1, output_dim = n_latent_factors_mf, name = "Video_Embedding_MF")(video_input)
	video_vec_mf = keras.layers.Flatten(name='FlattenVideos-MF')(video_embedding_mf)
	video_vec_mf = keras.layers.Dropout(0.4)(video_vec_mf)

	user_input = keras.layers.Input(shape=[1],name='User')

	user_embedding_mlp = keras.layers.Embedding( input_dim = n_users + 1, output_dim = n_latent_factor_users, name = "User-Embedding-MLP")(user_input)
	user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(user_embedding_mlp)
	user_vec_mlp = keras.layers.Dropout(0.4)(user_vec_mlp)

	user_embedding_mf = keras.layers.Embedding( input_dim = n_users + 1, output_dim = n_latent_factors_mf, name = "User-Embedding-MF")(user_input)
	user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(user_embedding_mf)
	user_vec_mf = keras.layers.Dropout(0.4)(user_vec_mf)


	concat = keras.layers.merge([video_vec_mlp, user_vec_mlp], mode='concat',name='CONCAT')
	concat_dropout = keras.layers.Dropout(0.4)(concat)
	dense = keras.layers.Dense(300,name='FullyConnected')(concat_dropout)
	dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
	dropout_1 = keras.layers.Dropout(0.4,name='Dropout-1')(dense_batch)
	dense_2 = keras.layers.Dense(150,name='FullyConnected-1')(dropout_1)
	dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)
	dropout_2 = keras.layers.Dropout(0.2,name='Dropout-2')(dense_batch_2)
	dense_3 = keras.layers.Dense(75,name='FullyConnected-2')(dropout_2)
	dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)
	pred_mlp  = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)

	pred_mf = keras.layers.merge([video_vec_mf, user_vec_mf], mode='dot',name='Dot')

	combine_mlp_mf = keras.layers.merge([pred_mf, pred_mlp], mode='concat',name='Concat-MF-MLP')
	result_combine = keras.layers.Dense(100,name='Combine-MF-MLP')(combine_mlp_mf)
	deep_combine = keras.layers.Dense(100,name='FullyConnected-4')(result_combine)


	result_inter = keras.layers.Dense(1,name='Prediction')(deep_combine)
	result = keras.activations.sigmoid(result_inter)*(5-0)+0 #this will allow to the model to converge faster as values is squashed between 0 and 5.

	model = keras.Model([user_input, video_input], result)
	opt = keras.optimizers.Adam(lr =0.01)
	model.compile(optimizer='adam',loss= 'mean_absolute_error')

	return model


Data = GetData()

n_latent_factor_users = 30
n_latent_factor_video = 50
n_latent_factors_mf	  = 10

model = Model_Recommendation(n_latent_factor_users,n_latent_factor_video,n_latent_factors_mf,Data)

model.summary()

history = model.fit([Data.UserID, Data.VideosID], Label, epochs=20, verbose=1, validation_split=0.1)

#predict the videos a user might like

user_id = 2314 # suppose user id is 2314 for which we like to recommend videos 
n_recommended = 50 # number of recommended videos we want to show to this user

recommended_list = Recommend_Videos(model,user_id,n_recommended)
print(recommended_list) #print the recommended list or some function like show() which displayes these videos to that user











