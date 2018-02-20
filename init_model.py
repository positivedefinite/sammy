from keras.models import model_from_json

def init(json_path,h5_path): 
	json_file = open(json_path,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights(h5_path)
	print("Loaded Model from disk")

	#compile and evaluate loaded model
#	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#	return loaded_model
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	#graph = tf.get_default_graph()

	#return loaded_model,graph
	return loaded_model