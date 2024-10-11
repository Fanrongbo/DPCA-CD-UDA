
from drawTool import plotFigureDA,plotFigureTarget
import pickle
import os
def save_pickle(data, file_name):
	f = open(file_name, "wb")
	pickle.dump(data, f)
	f.close()
def load_pickle(file_name):
	f = open(file_name, "rb+")
	data = pickle.load(f)
	f.close()
	return data


if __name__ == '__main__':
	# current_file_path = os.path.abspath(__file__)
	current_directory = os.path.dirname(os.path.abspath(__file__))

	figure_train_metrics = load_pickle(current_directory+"/../fig_train.pkl")
	figure_test_metrics = load_pickle(current_directory+"/../fig_val.pkl")
	figure_target_metrics = load_pickle(current_directory+"/../fig_T.pkl")

	# figure_train_metrics=np.load('fig_train.npy')
	# figure_test_metrics=np.load('fig_test.npy')
	num_epochs=len(figure_train_metrics['nochange_acc'])
	plotFigureDA(figure_train_metrics, figure_test_metrics, num_epochs,name='TrainLoad',model_type=None,time_now=None,load=True)
	num_epochs=len(figure_target_metrics['nochange_acc'])

	plotFigureTarget(figure_target_metrics,num_epochs,name='TargetLoad',model_type=None,time_now=None,load=True)

