# -*- coding: utf-8 -*-

# Stolen from https://github.com/spro/practical-pytorch

import random
import sys

import torch
import torch.nn as nn
from torch import optim

from model import TNNCell
from data_utils import *
from time_utils import *

MAX_LENGTH = 20
hidden_size = 1200
embedding_size = 30
total_iters = 40000
batch_size = 32
lr = 0.001
print_every = 200
noisy_rate = 0.4

def train(source_tensor, target_tensor, tnnCell, optimizer, criterion):

    optimizer.zero_grad()

    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    source_tensor = tnnCell.sourceEmbedding(source_tensor)
    target_tensor = tnnCell.targetEmbedding(target_tensor)

    input_xs = {i: source_tensor[i] for i in range(source_length)}
    output_ys = {}

    for ti in range(target_length):
        input_y = tnnCell.target_dict[SOS_token]
        output_xs = {}

        for si in range(0, source_length):
            output_xs[si], output_y = tnnCell(input_xs[si], input_y, noisy_rate=noisy_rate)
            input_y = output_y

        input_xs = output_xs
        output_ys[ti] = output_y

    x_loss = 0
    for si in output_xs:
        target_x = tnnCell.source_dict[SOS_token]
        x_loss += torch.sum(criterion(output_xs[si], target_x)) / batch_size
    x_loss = x_loss / len(output_xs)

    y_loss = 0
    for ti in output_ys:
        target_y = target_tensor[target_length-1-ti]
        y_loss += torch.sum(criterion(output_ys[ti], target_y)) / batch_size
    y_loss = y_loss / len(output_ys)

    loss = y_loss + x_loss
    loss.backward()
    optimizer.step()

    return loss


def sentenceFromTensor(output_y, partial_sentences, criterion):
    output_y = output_y.detach().repeat(target_lang.n_words, 1, 1)
    candidate_loss = torch.sum(criterion(output_y, tnnCell.target_dict), dim=2)
    minimum_loss, top_indices = torch.min(candidate_loss, dim=0)

    end_of_decoding = True
    for i in range(batch_size):
        topi = top_indices[i].item()
        if topi != SOS_token:
            end_of_decoding = False
        decoded_word = target_lang.index2word[topi]
        partial_sentences[i].insert(0, decoded_word)

    return end_of_decoding, partial_sentences


def evaluate(source_tensor, target_tensor, tnnCell, criterion, max_length):

    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    source_tensor = tnnCell.sourceEmbedding(source_tensor)
    target_tensor = tnnCell.targetEmbedding(target_tensor)

    partial_sentences = [[] for i in range(batch_size)]
    input_xs = {i: source_tensor[i] for i in range(source_length)}
    output_ys = {}

    for ti in range(max_length):
        output_xs = {}
        input_y = tnnCell.target_dict[SOS_token]

        for si in range(0, source_length):
            output_xs[si], output_y = tnnCell(input_xs[si], input_y, noisy_rate=0)
            input_y = output_y

        input_xs = output_xs
        output_ys[ti] = output_y

        end_of_decoding, partial_sentences = sentenceFromTensor(output_y, partial_sentences, criterion)
        if end_of_decoding:
            break

    x_loss = 0
    for si in output_xs:
        target_x = tnnCell.source_dict[SOS_token]
        x_loss += torch.sum(criterion(output_xs[si], target_x)) / batch_size
    x_loss = x_loss / len(input_xs)

    y_loss = 0
    for ti in output_ys:
        if ti < target_length:
            target_y = target_tensor[target_length-1-ti]
        else:
            target_y = tnnCell.target_dict[SOS_token]
        y_loss += torch.sum(criterion(output_ys[ti], target_y)) / batch_size
    y_loss = y_loss / len(output_xs)

    loss = y_loss + x_loss
    decoded_sentences = [''.join(decoded_words) for decoded_words in partial_sentences]

    return loss, decoded_sentences


def trainIters(tnnCell, n_iters, source_lang, target_lang, training_set, validation_set, print_every, learning_rate):
    print("batch size: ", batch_size, "learning_rate: ", lr)
    start = time.time()
    train_loss_total = 0  # Reset per print_every

    optimizer = optim.Adam(tnnCell.parameters(), lr=learning_rate)

    for iter in range(1, n_iters + 1):

        batch_pairs = [random.choice(training_set) for i in range(batch_size)]
        source_tensor, target_tensor = batchTensorsFromPairs(batch_pairs, source_lang, target_lang)

        if iter % print_every == 0:
            print("batch_pairs[0]: ", batch_pairs[0])
            print("")
        train_loss = train(source_tensor, target_tensor, tnnCell, optimizer, criterion)

        train_loss_total += train_loss

        if iter % print_every == 0:
            train_loss_avg = train_loss_total / print_every
            train_loss_total = 0
            print("training_set size: ", len(training_set))
            print('%s (%d %d%%) average loss %f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, train_loss_avg))
            print("")
            acc, loss = evaluateRandomly(tnnCell, training_set, source_lang, target_lang, batch_size * 10)
            print("Evaluated on training set loss %f accuracy %f\n" % (loss, acc))

            validation_acc, validation_loss = evaluateRandomly(tnnCell, validation_set, source_lang, target_lang, batch_size * 10)
            print("Evaluated on validation set loss %f accuracy %f\n" % (validation_loss, validation_acc))

            sys.stdout.flush()
            torch.save(tnnCell, "TNN_model_%s_%s.pt" % (source_name, target_name))
            if acc == 1.0 and train_loss_avg < 0.005:
                print("Stop training\n")
                sys.stdout.flush()
                break

def evaluateRandomly(tnnCell, dataset, source_lang, target_lang, n, randomly=True, max_length=MAX_LENGTH, info=False):

    total_loss = 0
    total = 0
    positive_count = 0
    batch_count = n // batch_size
    dataset = dataset[:batch_count*batch_size]
    for i in range(batch_count):
        total += batch_size
        if randomly:
            batch_pairs = [random.choice(dataset) for i in range(batch_size)]
        else:
            batch_pairs = dataset[i*batch_size:(i+1)*batch_size]
        source_tensor, target_tensor = batchTensorsFromPairs(batch_pairs, source_lang, target_lang)

        with torch.no_grad():
            evaluate_loss, decoded_sentences = evaluate(source_tensor, target_tensor, tnnCell, criterion, max_length)
        total_loss += evaluate_loss

        for j in range(batch_size):
            if batch_pairs[j][1] == decoded_sentences[j].replace(SOS, ''):
                if decoded_sentences[j][0] == SOS:
                    positive_count += 1
            if i == 0 and j < 3 or info:
                print('>', batch_pairs[j][0])
                print('=', batch_pairs[j][1])
                print('<', decoded_sentences[j][1:]) # skip leading SOS
                print('')

        accuracy = positive_count / total
        average_loss = total_loss / total
        if info:
            print("Exact match accuracy %f (%d / %d), average loss %f" % (accuracy, positive_count, total, average_loss))
            sys.stdout.flush()

    return accuracy, average_loss

source_name = "9"
target_name = "10"
source_lang, target_lang, pairs = prepareData(source_name, target_name, reverse=False)

short_samples = pairs[:-2000]
long_samples = pairs[-2000:]
random.shuffle(short_samples)
short_size = len(short_samples)
training_size = int(short_size * 0.8)
training_set = short_samples[:training_size]
validation_set = short_samples[training_size:]
testing_set = long_samples
print("training_set: ", random.choice(training_set))
print("validation_set: ", random.choice(validation_set))
print("testing_set: ", random.choice(testing_set))

criterion = nn.MSELoss(reduce=False)
tnnCell = TNNCell(source_lang.n_words, target_lang.n_words, hidden_size, embedding_size, batch_size)
trainIters(tnnCell, total_iters, source_lang, target_lang, training_set, validation_set, print_every, lr)

print("Evaluate on testing_set")
accuracy, loss = evaluateRandomly(tnnCell, testing_set, source_lang, target_lang, len(testing_set), randomly=False, max_length=2000, info=True)
print("loss %f accuracy %f" % (loss, accuracy))
