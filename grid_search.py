import numpy as np
from scipy.io import savemat
from classification import pre_process, train, evaluate_model

if __name__ == '__main__':

	model = "distilbert-base-uncased"
	models = [
		"distilbert-base-uncased",
		"bert-base-uncased",
		"bert-large-uncased",
		"bert-base-cased",
		"bert-large-cased",
		"roberta-base",
		"roberta-large"
	]
	device = "cuda"
	small_subset = False
	batch_size = 8



	for model in models:
		try:
			records = []
			for epochs in [5, 7, 9]:
				for lr in [1e-4, 5e-4, 1e-3]:
					pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(
						model, batch_size, device, small_subset)

					val_acc = evaluate_model(pretrained_model, validation_dataloader, device)['accuracy']
					test_acc = evaluate_model(pretrained_model, test_dataloader, device)['accuracy']
					records.append([epochs, lr, val_acc, test_acc])
			savemat(
				'%s.mat' % (model),
				{"records" : np.array(records)}
			)
		except:
			pass


