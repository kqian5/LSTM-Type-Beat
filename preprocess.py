import tensorflow as tf
import pretty_midi
import numpy as np


# only testing for one file rn
def get_data():
	np.set_printoptions(threshold=np.inf)
	midi_pretty_format = pretty_midi.PrettyMIDI('../maestro-v2.0.0/2018/2018midi (1).midi')
	piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
	piano_roll = piano_midi.get_piano_roll(fs=0.5)
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
		seq.append(curr_notes)
	print(seq[:50])
	return seq

get_data()

