from sklearn.preprocessing import LabelEncoder
import pickle

def label_encoder(training_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(training_labels)
    training_labels = label_encoder.transform(training_labels)

    with open('label_encoder.pickle','wb') as ecn_file:
        pickle.dump(label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    return training_labels
