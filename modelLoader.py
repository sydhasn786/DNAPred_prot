from sklearn.externals import joblib
from keras.models import load_model


class modelLoader:
	model = None

	def loadModel(self):
	    model = joblib.load('static/model.pkl')
	    return model

	def loadModel(self):
		model = load_model('static/saved_model.h5')
		return model
