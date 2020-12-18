import pandas as pd
import numpy as np

def createTestTrainSets(training_sentences, training_labels):
    d={'sentence':training_sentences,'label': training_labels}
    dataset = pd.DataFrame.from_dict(d)

    test_list = list(dataset.groupby(by='label',as_index=False).first()['sentence'])
    test_index = []
    for i,_ in enumerate(test_list):
        idx = dataset[dataset.sentence == test_list[i]].index[0]
        test_index.append(idx)

    train_index = [i for i in dataset.index if i not in test_index]

    training_set = dataset.loc[train_index]
    testing_set = dataset.loc[test_index]

    training_sentences = training_set['sentence']
    training_labels = training_set['label']

    testing_sentences = testing_set['sentence']
    testing_labels = testing_set['label']

    return training_sentences, training_labels, testing_sentences, testing_labels
