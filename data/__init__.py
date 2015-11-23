import utils

def get_n_cats(df):
	cats=df['Category']
	return max(cats)+1

def save_splited(out_path,af,extractor,prefix=".lb"):
    train_path=out_path.replace(prefix,"_train"+prefix)
    test_path=out_path.replace(prefix,"_test"+prefix)
    train,test=split_dataset(af)
    train_vectors=extractor(train)
    test_vectors=extractor(test)
    utils.to_labeled_file(train_path,train_vectors)
    utils.to_labeled_file(test_path,test_vectors)

def split_dataset(af):
    train=af[(af.Person % 2) ==0 ]
    test=af[(af.Person % 2) ==1 ]
    return train,test