import pandas as pd
import re, csv, torch
from nltk.corpus.reader.tagged import word_tokenize
import xml.etree.cElementTree as ET
import random
from sentence_transformers import SentenceTransformer, util, SentencesDataset, InputExample, losses, util, models, evaluation
from torch.utils.data import DataLoader
from itertools import islice
import os
from os.path import exists

MODEL = 'all-MiniLM-L6-v2-45ep'

#### HELPER FUNCTIONS: ####
# this function will take in a string , clean it up, and return it
def clean_text(text):
    token_words = re.sub("<.*?>|\\n|&quot;", " ", text.lower())
    token_words = word_tokenize(token_words)
    return " ".join(token_words)

# turn the xml into a dataframe
# df = pd.read_xml("./gdrive/MyDrive/IRCourse_Project_Files/Posts.xml")
# Function to convert XML file to Pandas Dataframe
# Credit: https://stackoverflow.com/questions/63286268/how-to-convert-a-large-xml-file-to-pandas-dataframe
def xml2df(file_path):
    # Parsing XML File and obtaining root
    tree = ET.parse(file_path)
    root = tree.getroot()

    dict_list = []

    for _, elem in ET.iterparse(file_path, events=("end",)):
        if elem.tag == "row":
            dict_list.append(elem.attrib)      # PARSE ALL ATTRIBUTES
            elem.clear()

    df = pd.DataFrame(dict_list)
    # Get rid of columns that we dont need
    df = df.drop(df.columns.difference(['Id', 'PostTypeId', 'AcceptedAnswerId', 'Body', 'Title']), axis=1)

    # fill in nulls for title
    df['Title'] = df['Title'].fillna('')
 
    # Clean the text of title and body
    df['Title'] = df['Title'].apply(clean_text)
    df['Body'] = df['Body'].apply(clean_text)
    return df

def createAnswerEmbeddings(df, model):
    return model.encode(df['Body'].tolist(), convert_to_tensor=True)

def callback(score, epoch, steps):
    # This method is used to write the loss for each epoch on the validation set
    csv_writer_Epochs.writerow([score, epoch, steps])

def createTriples():
    # get a df of questions, get rid of questions without accepted answers, get rid of queries
    questions_df = df[df['PostTypeId'] == 1]
    exclude_ids = [127968, 67284, 1232, 18375, 47604, 8002, 51721, 11033, 4317, 1, 
                3548, 71, 73945, 2291, 98549, 67032, 66988, 59563, 105388, 120149]
    # Remove the query ids from the questions dataframe
    questions_df = questions_df[~questions_df["Id"].isin(exclude_ids)]
    questions_df['Id'] = questions_df['Id'].apply(int)
    questions_df = questions_df.drop(['Body'], axis=1)
    # Get negative and positive questions
    positive_questions_df = questions_df[questions_df['Id'] % 2 == 0]
    negative_questions_df = questions_df[questions_df['Id'] % 2 == 1]
    # Get a dataframe of just answers
    answers_df = df[df['PostTypeId'] == 2].drop(['AcceptedAnswerId', 'Title'], axis=1)
    # Get the accepted answer for every question that has one and add it to the positive list
    positive_questions_merge_df = positive_questions_df.merge(answers_df, how='left', left_on='AcceptedAnswerId', right_on='Id')
    positive_list = positive_questions_merge_df.drop(positive_questions_merge_df.columns.difference(['Title', 'Body']), axis=1).values.tolist()
    # Positive list: {question_id, accepted_answer, 1}
    for value in positive_list:
      value.append(1)
    answers_id_list_temp = answers_df.drop(answers_df.columns.difference(['Id']), axis=1).values.tolist()
    answers_id_list = []
    for value in answers_id_list_temp:
      answers_id_list.append(value[0])
    # Negative list: {question_id, random_answer, 0}
    negative_questions_df['RandomAnswersId'] = ''
    for ind in negative_questions_df.index:
      negative_questions_df['RandomAnswersId'][ind] = random.choice(answers_id_list)
    negative_questions_merge_df = negative_questions_df.merge(answers_df, how='left', left_on='RandomAnswersId', right_on='Id')
    negative_list = negative_questions_merge_df.drop(negative_questions_merge_df.columns.difference(['Title', 'Body']), axis=1).values.tolist()
    for value in negative_list:
      value.append(0)
    return {0: positive_list, 1: negative_list}

""" Copied and slightly modified from Behrooz Mansouri from SentenceBERTFineTuning.py"""
def split_data(data, split):
    # takes in list as the input data and return split of data (splitting into 10 pieces)
    length = int(len(data) / split)  # length of each fold
    pieces = []
    for i in range((split-1)):
        pieces.append(data[i * length: (i + 1) * length])
    pieces.append(data[(split-1) * length:len(data)])
    return pieces

def finetuneModel():
    pos_neg = createTriples()
    print(pos_neg[0][0])
    print(pos_neg[1][0])
    
    # Defining necessary variables for loss functions
    train_samples_MNRL = []

    # Lists used for the validation set of triplets {query, answer, status} (i.e. {"how to tame a dragon", "steel is the best material", 0} for negative answer)
    # query = all queries used for validation set
    # candidate = all answers to compare query to in validation set
    # status = status of whether query to answer is a postiive or negative answer
    evaluator_query = []
    evaluator_candidate = []
    evaluator_status = []

    # Parameters for fine-tuning
    num_epochs = 45
    train_batch_size = 64
    model.max_seq_length = 256

    # For both the negative and positive sets:
    for label in pos_neg:
        # Pull either negative or positive set
        instance = pos_neg[label]
        # Randomize and split set
        random.shuffle(instance)
        splits = split_data(instance, 10) # Returns: [[[qid, cand, label], [qid, cand, label],...], [[qid, cand, label]], [qid, cand, label],...], ...]
        # Divide portions into validation and training sets
        validation = splits[-1] # Only take last batch
        training = splits[:-1] # Training is every batch but the last 

        # For each batch in training:
        for split in training:
            for triples in split:
                query = triples[0]
                candidate = triples[1]
                label = triples[2]
                train_samples_MNRL.append(InputExample(texts=[query, candidate], label=label))
        # Save validation triplets to evaluator lists
        for triples in validation:
            evaluator_query.append(triples[0])
            evaluator_candidate.append(triples[1])
            evaluator_status.append(triples[2])

    # Prep MultipleNegativesRankingLoss:
    train_dataset_MNRL = SentencesDataset(train_samples_MNRL, model=model)
    train_dataloader_MNRL = DataLoader(train_dataset_MNRL, shuffle=True, batch_size=train_batch_size)
    train_loss_MNRL = losses.MultipleNegativesRankingLoss(model)

    # Prep validation set:
    evaluator = evaluation.EmbeddingSimilarityEvaluator(evaluator_query, evaluator_candidate, evaluator_status, write_csv="evaluation_epoch.csv")

    # Fine-Tune the model:
    epoch_csv_file = "./data/epochs.tsv"
    post_xml_file = "Posts.xml"
    with open(epoch_csv_file, mode='w', newline='') as csv_file:
        csv_writer_Epochs = csv.writer(csv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        model.fit(
            train_objectives=[(train_dataloader_MNRL, train_loss_MNRL)],
            evaluator=evaluator,
            epochs=num_epochs,
            warmup_steps=1000,
            output_path=MODEL,
            show_progress_bar=True,
            callback=callback
        )

def printTopK(results, k=5):
    print_list = dict(islice(sorted(results.items(), key=lambda item: item[1], reverse=True), k))
    for id in print_list:
        print(str(id) + "\t\t" + str(print_list[id]))

def makeUserQuery():
    userInput = "Y"
    final_dict = {}
    while(userInput[0].upper() == 'Y'):
        query_results = {}
        query = input("User Query: ")
        query_embedding = model.encode(query, convert_to_tensor=True)               # Create embedding of query.
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1000) # Make semantic search
        hits = hits[0]
        for hit in hits:
            index = hit['corpus_id']
            answer_id = answer_ids[index]
            score = hit['score']
            query_results[answer_id] = score
        final_dict[query] = query_results
        printTopK(query_results, 10)
        userInput = input("Make another query? (Y/N): ")
        while(userInput[0].upper() != 'N' and userInput[0].upper() != 'Y'):
            userInput = input("Make another query? (Y/N): ")
    return final_dict

def retrieveQueryResults(queries):
    final_dict = {}
    for topic_id in queries:
        query_results = {}
        query = queries[topic_id]
        query_embedding = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1000)
        hits = hits[0]  # Get the hits for the first query
        for hit in hits:
            index = hit['corpus_id']
            answer_id = answer_ids[index]
            score = hit['score']
            query_results[answer_id] = score
        print(queries[topic_id])
        printTopK(query_results, 5)
        final_dict[topic_id] = query_results
    return final_dict

def createRunFile(filename, results):
    qid_start = "Q00"
    count = 1
    with open("./data/" + filename+ ".tsv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for topic_id in results:
            if count == 10:
                qid_start = "Q0"
            qid_combine = qid_start + str(count)
            count += 1
            result_map = results[topic_id]
            result_map = dict(sorted(result_map.items(), key=lambda item: item[1], reverse=True))
            rank = 1
            for post_id in result_map:
                score = result_map[post_id]
                csv_writer.writerow([qid_combine, "Q0",  post_id, str(rank), str(score), MODEL])
                rank += 1
                if rank > 1000:
                    break

#### METHOD CALLS: ####
# Ask user if they need to do initial preprocessing; alternative is loading from ./data
validInput = False
userInput = input("1. Load from data. (Creates any missing assets)\n2. Create new system. \nUser choice: ")
while validInput == False: 
    if(userInput[0].upper() == '1'):
        # On 1: create or read the posts dataframe
        if(os.path.exists("./data/Posts_DF.csv")):
            print("Loading Posts_DF.csv...")
            df = pd.read_csv("./data/Posts_DF.csv")
        else:
            print("Parsing Posts.xml...")
            df = xml2df("./data/Posts.xml")                            # Generates inverted index with count from Posts.xml
            df.to_csv("./data/Posts_DF.csv", index=False)              # Saves inverted index in TSV form
            df = pd.read_csv("./data/Posts_DF.csv")
        # Model loading / Finetuning:
        if(os.path.exists("./data/Finetuned_Model")):
            print("Loading Finetuned_Model...")
            model = torch.load("./data/Finetuned_Model")
            finetuneModel()
        else:
            print("Loading \"all-MiniLM-L6-v2\"...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Finetuning the model...")
            finetuneModel()
            torch.save(model, "./data/Finetuned_Model")

        # Create embeddings for just the answers
        if(os.path.exists("./data/Corpus_Embeddings.pt")):
            print("Loading Corpus_Embeddings...")
            corpus_embeddings = torch.load("./data/Corpus_Embeddings.pt")
        else:
            print("Creating posts embeddings...")
            corpus_embeddings = createAnswerEmbeddings(df[df['PostTypeId'] == '2'], model)
            torch.save(corpus_embeddings, "./data/Corpus_Embeddings.pt") # Save the embeddings
        # Get answer_ids
        answer_ids = df[df['PostTypeId'] == 2]['Id'].tolist()
        validInput = True
        print("Done.")
    elif(userInput[0].upper() == '2'):
        # On 2: Create new system; get posts df:
        print("Parsing Posts.xml...")
        df = xml2df("./data/Posts.xml")                            # Generates inverted index with count from Posts.xml
        df.to_csv("./data/Posts_DF.csv", index=False)              # Saves inverted index in TSV form
        df = pd.read_csv("./data/Posts_DF.csv")
        # Create model and finetune it:
        print("Loading \"all-MiniLM-L6-v2\"...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Finetuning the model...")
        finetuneModel()
        torch.save(model, "./data/Finetuned_Model")
        # Create Corpus embeddings
        print("Creating posts embeddings...")
        corpus_embeddings = createAnswerEmbeddings(df[df['PostTypeId'] == '2'], model)
        torch.save(corpus_embeddings, "./data/Corpus_Embeddings.pt") # Save the embeddings
        # Get answer_ids
        answer_ids = df[df['PostTypeId'] == 2]['Id'].tolist()
        validInput = True
        print("Done.")
    else:
        print("Invalid input. Input 1 or 2")
        userInput = input("1. Load from data. (Creates any missing assets)\n2. Create new system. \nUser choice: ")


# Given our DF, model and embeddings, get queries:
final_dict = {}
validInput = False
userInput = input("Input user query? (Y/N): ")
while validInput == False: 
    if(userInput[0].upper() == 'Y'):
        final_dict = makeUserQuery()
        validInput = True
    elif(userInput[0].upper() == 'N'):
        # On no: Use the set of queries selected by ourselves
        query_ids = [127968, 67284, 1232, 18375, 47604, 8002, 51721, 11033, 4317, 1, 
               3548, 71, 73945, 2291, 98549, 67032, 66988, 59563, 105388, 120149]
        # create a dictionary of {query_id: query_title}
        queries = {}
        for id in query_ids:
            queries[id] = df[df['Id'] == id]['Title'].values[0]
        final_dict = retrieveQueryResults(queries)
        validInput = True
    else:
        print("Invalid input. Input Y or N")
        userInput = input("Input user query? (Y/N): ")

# Ask user if they want to save the results to a Run file
validInput = False
userInput = input("Save results to Run file? (Y/N): ")
while validInput == False: 
    if(userInput[0].upper() == 'Y'):
        filename = input("Filename: ")
        createRunFile(filename, final_dict)
        validInput = True
    elif(userInput[0].upper() == 'N'):
        print("Closing retrieval system.")
        validInput = True
    else:
        print("Invalid input. Input Y or N")
        userInput = input("Input user query? (Y/N): ")
