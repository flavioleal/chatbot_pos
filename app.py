from flask import Flask, render_template, request
import pickle
import json
import random
import tensorflow
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


app = Flask(__name__)

with open("intents.json", 'rb') as file:
    data = json.load(file)

    # try:
    #     with open("data.pickle",  "rb") as f:
    #         words, words_size, labels,  training, output = pickle.load(f)
    # except:
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

words_size = len(words)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

        # with open("data.pickle",  "wb") as f:
        #     pickle.dump((words, words_size, labels, training, output), f)


tensorflow.reset_default_graph()
tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.embedding(net, input_dim=words_size, output_dim=128)
#net = tflearn.lstm(net, 128, dropout=0.8, return_seq=True)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

#try:
#   model.load("model.tflearn")
#except:
model = tflearn.DNN(net)
history = model.fit(training, output, n_epoch=400,
                        batch_size=128, show_metric=True, run_id='intents')
    #model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(userText):
    #print("Comece a conversar com o bot (digite quit para parar)!")

    # if userText.lower() == "quit":
    #    break

    results = model.predict([bag_of_words(userText, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        return str(random.choice(responses))
    else:
        return str("Não entendi, tente novamente. ")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chat(userText))


if __name__ == "__main__":
    app.run()
