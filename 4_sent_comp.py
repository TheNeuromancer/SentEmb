from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse
import random

import utils

from copy import deepcopy
from ipdb import set_trace


# -----------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


# -----------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------
    # Settings
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sent-file", type=str, default="./tenenbaum_sentences.txt", help="path to sentences to consider"
    )

    parser.add_argument(
        "--batch_size", default=64, type=int, help="batch size for extracting features."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained language models. (default: 'bert-base-uncased')",
    )
    parser.add_argument(
        "--embed_method",
        type=str,
        default="ave_last_hidden",
        help="Choice of method to obtain embeddings (default: 'ave_last_hidden')",
    )
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=2,
        help="Topological Embedding Context Window Size (default: 2)",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=4,
        help="Starting layer for fusion (default: 4)",
    )
    parser.add_argument(
        "--tasks", type=str, default="sts", help="choice of tasks to evaluate on"
    )
    args = parser.parse_args()


    # -----------------------------------------------
    # Set seed
    set_seed(args)
    # Set up logger
    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

    # -----------------------------------------------
    # Set Model
    params = vars(args)

    if args.model_type == "USE":
        import tensorflow_hub as hub
        import tensorflow as tf
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    else:
        import torch
        from transformers import *  
        torch.manual_seed(args.seed)
        # -----------------------------------------------
        # Set device
        # torch.cuda.set_device(-1)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        args.device = device
        config = AutoConfig.from_pretrained(params["model_type"], cache_dir="./cache")
        config.output_hidden_states = True
        tokenizer = AutoTokenizer.from_pretrained(params["model_type"], cache_dir="./cache")
        model = AutoModelWithLMHead.from_pretrained(
            params["model_type"], config=config, cache_dir="./cache"
        )
        model.to(params["device"])

    # -----------------------------------------------
    sentences_O = []
    with open(args.sent_file, "r") as file:
        for line in file.readlines():
            sentences_O.append(line.rstrip())

    # sentences_O = [ "a young woman in front of an old man",
    #                 "a black cat in front of a white dog",
    #                 "a short woman in front of a tall man",
    #                 "a small bug on a large flower",
    #                 "a small book on a black table",
    #                 "a young girl in front of a happy soldier",
    #                 "a black cow in front of a brown horse",
    #                 "a sleeping boy in front of a smiling woman",
    #                 "a white dog on a brown chair",
    #                 "a happy man in front of an old woman",
    #                 "a young doctor in front of a smiling patient",
    #                 "a red apple on green paper",
    #                 "a white plate on a blue pillow",
    #                 "a blue pen on a red folder",
    #                 "a green pear on a brown leaf",
    #                 "a yellow banana on a green knife",
    #                 "an orange pepper on a yellow folder",
    #                 "a plastic bag in front of a brown bottle",
    #                 "a brown frog on green grass",
    #                 "a black magazine in front of a white mug",
    #                 "a pink bowl in front of a blue cup",
    #                 "a tissue box on yellow paper",
    #                 "a purple shirt on a green knife",
    #                 "a young man in front of an angry woman",
    #                 "a black phone on gray pants",
    #                 "a rusty bicycle in front of an old fence",
    #                 "a black marker on a red shirt",
    #                 "a white sock on black headphones",
    #                 "an open book in front of a closed window",
    #                 "a full glass in front of an empty bottle"]

    sentences_N = []
    for sent in sentences_O:
        sent_N = deepcopy(sent)
        sent_N = sent_N.split()
        sent_N[2], sent_N[-1] = sent_N[-1], sent_N[2]
        sentences_N.append(' '.join(sent_N))

    sentences_A = []
    for sent in sentences_O:
        sent_A = deepcopy(sent)
        sent_A = sent_A.split()
        sent_A[0:2], sent_A[-3:-1] = sent_A[-3:-1], sent_A[0:2]
        sentences_A.append(' '.join(sent_A))

    sentences_P = []
    for sent in sentences_O:
        sent = sent.replace('in front of', 'behind')
        sent = sent.replace(' above ', ' below ') # for enriched_sentences.txt
        sent = sent.replace(' on ', ' below ') # for tenenbaum_sentences.txt (original ones, not the best replacement ever ... but we don't know what they used in the paper)
        # ACTIVE SENTENCES
        sent = sent.replace(' greets ', ' is greeted by ')
        sent = sent.replace(' kicks ', ' is kicked by ')
        sent = sent.replace(' likes ', ' is liked by ')
        sent = sent.replace(' loves ', ' is loved by ')
        sent = sent.replace(' hates ', ' is hated by ')
        sent = sent.replace(' knows ', ' is known by ')
        sent = sent.replace(' tells ', ' is told by ')
        sent = sent.replace(' needs ', ' is needed by ')
        sent = sent.replace(' helps ', ' is helped by ')
        sent = sent.replace(' believes ', ' is believed by ')
        sent = sent.replace(' hears ', ' is heard by ')
        sent = sent.replace(' remembers ', ' is remembered by ')
        sent = sent.replace(' serves ', ' is served by ')
        sent = sent.replace(' kills ', ' is killed by ')
        sent = sent.replace(' reports ', ' is reported by ')
        sent = sent.replace(' greet ', ' are greeted by ')
        sent = sent.replace(' kick ', ' are kicked by ')
        sent = sent.replace(' like ', ' are liked by ')
        sent = sent.replace(' love ', ' are loved by ')
        sent = sent.replace(' hate ', ' are hated by ')
        sent = sent.replace(' know ', ' are known by ')
        sent = sent.replace(' tell ', ' are told by ')
        sent = sent.replace(' need ', ' are needed by ')
        sent = sent.replace(' help ', ' are helped by ')
        sent = sent.replace(' believe ', ' are believed by ')
        sent = sent.replace(' hear ', ' are heard by ')
        sent = sent.replace(' remember ', ' are remembered by ')
        sent = sent.replace(' serve ', ' are served by ')
        sent = sent.replace(' kill ', ' are killed by ')
        sent = sent.replace(' report ', ' are reported by ')
        sentences_P.append(sent)

    sentences_M = []
    for sent in sentences_P:
        sent_M = sent.lower().split()
        sent_M[0:3], sent_M[-3::] = sent_M[-3::], sent_M[0:3]
        # sent_M[0] = sent_M[0][1::] + sent_M[0][0]
        sentences_M.append(' '.join(sent_M))

    # set_trace()
    # print(sentences_O[0])
    # print(sentences_N[0])
    # print(sentences_A[0])
    # print(sentences_P[0])
    # print(sentences_M[0])

    all_sentences = [[o, n, a, p, m] for o, n, a, p, m in zip(sentences_O, sentences_N, sentences_A, sentences_P, sentences_M)]
    for sents in all_sentences: print(sents)
    exit()

    ranks = []


    for sentences in all_sentences:

        if args.model_type == "USE":
            # -----------------------------------------------
            embedding = embed(sentences)

            similarity_N = (tf.tensordot(embedding[0], embedding[1], axes=1) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[1]))
            similarity_A = (tf.tensordot(embedding[0], embedding[2], axes=1) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[2]))
            similarity_P = (tf.tensordot(embedding[0], embedding[3], axes=1) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[3]))
            similarity_M = (tf.tensordot(embedding[0], embedding[4], axes=1) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[4]))

        else: # all BERT based models
            # -----------------------------------------------
            sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
            features_input_ids = []
            features_mask = []
            for sent_ids in sentences_index:
                # Truncate if too long
                if len(sent_ids) > params["max_seq_length"]:
                    sent_ids = sent_ids[: params["max_seq_length"]]
                sent_mask = [1] * len(sent_ids)
                # Padding
                padding_length = params["max_seq_length"] - len(sent_ids)
                sent_ids += [0] * padding_length
                sent_mask += [0] * padding_length
                # Length Check
                assert len(sent_ids) == params["max_seq_length"]
                assert len(sent_mask) == params["max_seq_length"]

                features_input_ids.append(sent_ids)
                features_mask.append(sent_mask)

            features_mask = np.array(features_mask)

            batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
            batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
            batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            model.zero_grad()

            with torch.no_grad():
                features = model(**inputs)[1]

            # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
            # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
            all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

            embed_method = utils.generate_embedding(params["embed_method"], features_mask)
            embedding = embed_method.embed(params, all_layer_embedding)

            similarity_N = (embedding[0].dot(embedding[1]) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[1]))
            similarity_A = (embedding[0].dot(embedding[2]) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[2]))
            similarity_P = (embedding[0].dot(embedding[3]) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[3]))
            similarity_M = (embedding[0].dot(embedding[4]) / np.linalg.norm(embedding[0]) / np.linalg.norm(embedding[4]))

        # print("The similarity with noun change sentence is:", similarity_N)
        # print("The similarity with adjective change sentence is:", similarity_A)
        # print("The similarity with preposition change sentence is:", similarity_P)
        # print("The similarity with meaning preservation change sentence is:", similarity_M)
        # print('\n')

        ranks.append(4 - np.argsort([similarity_N, similarity_A, similarity_P, similarity_M]))

print(np.mean(ranks, 0))
# set_trace()
