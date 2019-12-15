import tensorflow as tf
import numpy as np
from preprocess import process
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Activation
import pretty_midi


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        super(Model, self).__init__()

        # set hyperparameters
        self.vocab_size = vocab_size
        self.embedding_size = 40
        self.window_size = 50
        self.batch_size = 50
        self.optimizer = tf.keras.optimizers.Adam()

        # establish model and layers
        model = tf.keras.Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        self.model = model


    def call(self, inputs):
        return self.model(inputs)
        
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

def train(model, train_inputs, train_labels):
    # train on batches
    for i in range(0, len(train_inputs), model.batch_size * model.window_size):
        with tf.GradientTape() as tape:
            if i + model.batch_size * model.window_size > len(train_inputs):
                break
            inputs = np.array(train_inputs[i:i+(model.batch_size*model.window_size)]).reshape((model.batch_size, model.window_size))
            labels = np.array(train_labels[i:i+(model.batch_size*model.window_size)]).reshape((model.batch_size, model.window_size))
            logits = model.call(inputs)
            loss = model.loss(logits, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_music(note1, length, vocab, model):
    # when given a starting note, predict music for a certain length
    reverse_vocab = {idx:chord for chord, idx in vocab.items()}

    first_note = note1
    first_note_index = vocab[note1]
    next_input = np.zeros(shape=(1, 1))
    next_input[0][0] = first_note_index
    song = [first_note]

    for i in range(length):
        p = np.random.uniform()
        logits = model.call(next_input).numpy()
        if p > 0.7:
            out_index = np.argmax(np.array(logits[0][0]))
        else:
            logits[0][0][0] += 1 - sum(logits[0][0])
            out_index = np.random.choice(list(vocab.values()), p=logits[0][0])
        song.append(reverse_vocab[out_index])
        next_input = np.array([np.array([out_index])])
    return song

def convert_to_midi(inputs, f, output_fs):
    # convert model output back to midi file
    array_piano_roll = np.zeros((128, 1000), dtype=np.int16)
    for t, chord in enumerate(inputs):
        if chord =="unk":
            continue
        else:
            notes = chord.split(",")
            for n in notes:
                array_piano_roll[int(n), t] = 1
    midi = piano_roll_to_pretty_midi(array_piano_roll, fs=output_fs)
    for note in midi.instruments[0].notes:
        note.velocity = np.random.randint(low=50, high=120)
    midi.write(f)

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm



def main():
    # preprocess
    data, vocab = process()

    train_inputs = data[0:len(data)-1]
    train_labels = data[1:len(data)]

    # train model
    model = Model(len(vocab))

    train(model, train_inputs, train_labels)
    reverse_vocab = {idx:chord for chord, idx in vocab.items()}

    # generate output
    num_outputs = 10
    output_fs = 3
    for i in range(num_outputs):
        seed = np.random.randint(low=1, high=len(vocab) - 1)
        convert_to_midi(generate_music(str(reverse_vocab[seed]), 200, vocab, model), "result" + str(i + 1) + ".mid", output_fs)

if __name__ == '__main__':
   main()
