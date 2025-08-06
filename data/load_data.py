from utils.prepare_data import prepare_data

train_file_path = './train_data.xlsx'
validation_file_path = './validation_data.xlsx'
test_file_path = './test_data.xlsx'

def load_data(time_step):
    return prepare_data(train_file_path, test_file_path, validation_file_path, time_step)
