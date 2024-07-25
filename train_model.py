import numpy as np
from mynn.optimizers.sgd import SGD
import mygrad.nnet

from ___ import Image2Caption # Model

def cos_sim(x, y):
    out = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return out

def accuracy(predictions, truth): # i just reused the old one, and i think it should be fine, except maybe the == in line 14?
    predicted_labels = np.argmax(predictions, axis = 1)
    
    prediction_vs_truth = np.array(predicted_labels == truth)

    fraction_correct = np.mean(prediction_vs_truth)
    
    return fraction_correct

def train_model(train_image_descriptors, good_image_embeddings, bad_image_embeddings):

    model = Image2Caption()
    optim = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
    margin = 0.1

    batch_size = 32

    for epoch_cnt in range(180): # revise epoch count
        idxs = np.arange(len(train_image_descriptors)) # train_image_IDs is all the image IDs set aside for training data
        np.random.shuffle(idxs)  
        
        for batch_cnt in range(0, len(train_image_descriptors) // batch_size):
            batch_indices = idxs[(batch_cnt * batch_size):((batch_cnt + 1) * batch_size)]
            batch = train_image_descriptors[batch_indices]
            
            outputs = model(batch)

            sim_to_good = cos_sim(outputs, good_image_embeddings)
            sim_to_bad = cos_sim(outputs, bad_image_embeddings)

            loss = mygrad.nnet.margin_ranking_loss(sim_to_good, sim_to_bad, 1, margin)

            loss.backward()
            optim.step()

            acc = accuracy(outputs, good_image_embeddings)

    return acc
