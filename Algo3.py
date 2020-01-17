def GetData():
	Construct_CSV # it contains various rows of UserID, VideoID and Label
	return CSVTable

def GetVideos(user_id):
	Sample_Videos_User_has_Not_Seen_withIDEqualsUser_id
	return VideoID_List

def Recommend_Videos(model,user_id,user_meta,n_recommended):
	VideoID_List = GetVideos(user_id)

	y_pred = np.round(model.predict([user_meta, user_id, VideoID_List.video, VideoID_List.meta]),0) # get the score for unseen videos of this user
	arr = sorted(range(len(y_pred)), key=lambda k: y_pred[k], reverse=True) #return indices of items which are sorted according to higest values
	
	recommended_list = [] 
	for i in arr: 
		recommended_list.append(VideoID_List[i])

	return recommended_list[:n_recommended] # return the video ID of first "n_recommended" recommendations 

def Model_Recommendation(Data):
	
	n_users = len( Data.UserID.Unique())	#get number of unique users
	n_videos = len( Data.VideosID.Unique())	#get number of unique videos


	user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, 10)(user)

    y = len(vector_for_user_description)#this is user's meta
    user_meta = Input(shape=(y,))
    u_m = EmbeddingLayer(n_users, 20)(user_meta)
    
    video = Input(shape=(1,))
    v = EmbeddingLayer(n_videos, 50)(video)

    x = len(vector_for_video_description)#these vectors can be generated in a similar manner which is mentioned in Algorithm 1
    video_meta = Input(shape=(x,))
    v_m = EmbeddingLayer(n_videos, 30)(video_meta)

    z = merge([u_m, u, v, v_m], mode='concat')
    z = Dropout(0.05)(z)
    
    z = Dense(10, kernel_initializer='he_normal')(z)
    z = Activation('relu')(z)
    z = Dropout(0.5)(z)
    
    z = Dense(1, kernel_initializer='he_normal')(z)
    z = Activation('sigmoid')(z)*(5-0)+0

    model = Model(inputs=[user_meta, user, video, video_meta], outputs=z)
    opt = Adam(lr =0.01)
	model.compile(optimizer='adam',loss= 'mean_absolute_error')

	return model



Data = GetData()

model = Model_Recommendation(Data)

model.summary()

history = model.fit([Data.UserMeta, UserID, Data.Videos, Data.VideoMeta], Label, epochs=20, verbose=1, validation_split=0.1)

#predict the videos a user might like

user_id = 2314 # suppose user id is 2314 for which we like to recommend videos 
n_recommended = 50 # number of recommended videos we want to show to this user

recommended_list = Recommend_Videos(model,user_id,user_meta,n_recommended)
print(recommended_list) #print the recommended list or some function like show() which displayes these videos to that user











