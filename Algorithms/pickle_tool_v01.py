import pickle

def save_list_by_pickle(input_list, target_path):
  with open(target_path, 'wb') as f:
    pickle.dump(input_list, f)

def load_list_by_pickle(source_path):
  recieved_list = []
  with open(source_path, 'rb') as f:
    recieved_list = pickle.load(f)
  return recieved_list
