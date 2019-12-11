import tensorflow as tf
import pretty_midi
import numpy as np
import sys
import pickle

# only testing for one file rn
def get_data(f):
	np.set_printoptions(threshold=np.inf)
	midi_pretty_format = pretty_midi.PrettyMIDI(f)
	piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
	piano_roll = piano_midi.get_piano_roll(fs=2)
	seq = []

	for i in range(len(piano_roll[0])):
		curr_notes = ''
		for j in range(len(piano_roll)):
			if piano_roll[j][i] != 0:
				if not curr_notes:
					curr_notes = str(j)
				else:
					curr_notes = curr_notes + ',' + str(j)
		if not curr_notes:
			curr_notes = 'unk'
		# if curr_notes == '':
		# 	continue
		seq.append(curr_notes)
	# print(seq[:50])
	return seq


def process():
	print("preprocessing started")
	sys.stdout.flush()
	data = []
	for i in range(1, 99):
		data.extend(get_data('../maestro-v2.0.0/2018/2018midi (' + str(i) + ').midi'))
	# for i in range(1, 140):
	# 	data.extend(get_data('../maestro-v2.0.0/2017/2017midi (' + str(i) + ').midi'))
	# for i in range(1, 129):
	# 	data.extend(get_data('../maestro-v2.0.0/2015/2015midi (' + str(i) + ').midi'))
	# for i in range(1, 105):
	# 	data.extend(get_data('../maestro-v2.0.0/2014/2014midi (' + str(i) + ').midi'))
	# for i in range(1, 127):
	# 	data.extend(get_data('../maestro-v2.0.0/2013/2013midi (' + str(i) + ').midi'))
	# for i in range(1, 163):
	# 	data.extend(get_data('../maestro-v2.0.0/2011/2011midi (' + str(i) + ').midi'))
	# for i in range(1, 125):
	# 	data.extend(get_data('../maestro-v2.0.0/2009/2009midi (' + str(i) + ').midi'))
	# for i in range(1, 147):
	# 	data.extend(get_data('../maestro-v2.0.0/2008/2008midi (' + str(i) + ').midi'))
	# for i in range(1, 115):
	# 	data.extend(get_data('../maestro-v2.0.0/2006/2006midi (' + str(i) + ').midi'))
	# for i in range(1, 132):
	# 	data.extend(get_data('../maestro-v2.0.0/2004/2004midi (' + str(i) + ').midi'))

	# with open('data.pkl', 'wb') as f:
	# 	pickle.dump(data, f)
	# with open('data.pkl', 'rb') as f:
	# 	data = pickle.load(f)
	vocab = {n:i for i, n in enumerate(list(set(data)))}
	tokenized = []
	for chord in data:
		tokenized.append(vocab[chord])
	print("preprocessing complete")
	# with open('tokenized.pkl', 'rb') as f1:
	# 	tokenized = pickle.load(f1)
	# f1.close()
	# with open('vocab.pkl', 'rb') as f2:
	# 	vocab = pickle.load(f2)
	# f2.close()
	return tokenized, vocab

# print(get_data('../maestro-v2.0.0/2018/2018midi (1).midi'))


