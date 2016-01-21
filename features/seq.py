import utils

def parse_seq(filename):
	seqs=utils.read_file(filename)
	X=[seq_i.split("#")[0] for seq_i in seqs]
	#X=[to_ngram(x_i) for x_i in seqs]
	y=[seq_i.split("#")[1] for seq_i in seqs]
	y=[int(y_i) for y_i in y]
	print(y)

def to_ngram(seq):
    ngrams=[]
    for i in range(0,len(seq)-1):
        ngrams.append(seq[i]+seq[i+1])
    return ngrams 