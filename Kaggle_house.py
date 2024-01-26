import pandas
import Kaggle_download
from Kaggle_download import DATA_HUB

DATA_HUB['kaggle_house_train'] = (Kaggle_download.DATA_URL + 'kaggle_house_pred_train.csv','585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (Kaggle_download.DATA_URL + 'kaggle_house_pred_test.csv','fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pandas.read_csv(Kaggle_download.download('kaggle_house_train'))
test_data = pandas.read_csv(Kaggle_download.download('kaggle_house_test'))

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])