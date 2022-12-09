import pandas as pd
import re, csv, torch
from nltk.corpus.reader.tagged import word_tokenize
import xml.etree.cElementTree as ET
from sentence_transformers import SentenceTransformer, util
from itertools import islice

MODEL = 'all-MiniLM-L6-v2'

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
  return df

def createAnswerEmbeddings(df, model):
    return model.encode(df['Body'].tolist(), convert_to_tensor=True)

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
userInput = input("First time processing? (Y/N): ")
while validInput == False: 
    if(userInput[0].upper() == 'Y'):
        # On yes: create the dataframe
        print("Parsing Posts.xml...")
        df = xml2df("./data/Posts.xml")                            # Generates inverted index with count from Posts.xml
        df.to_csv("./data/Posts_DF.csv", index=False)              # Saves inverted index in TSV form
        print("Loading the model...")
        model = SentenceTransformer(MODEL)              # Load the model
        torch.save(model, "./data/Pretrained_Model")
        # Create embeddings for just the answers
        print("Creating posts embeddings...")
        corpus_embeddings = createAnswerEmbeddings(df[df['PostTypeId'] == '2'], model)
        torch.save(corpus_embeddings, "./data/Corpus_Embeddings.pt") # Save the embeddings
        answer_ids = df[df['PostTypeId'] == 2]['Id'].tolist()
        validInput = True
        print("Done.")
    elif(userInput[0].upper() == 'N'):
        # On no: Load dataframe, model and embeddings
        print("Loading Posts_DF.csv...")
        df = pd.read_csv("./data/Posts_DF.csv")                         # Load Posts DF
        print("Loading retrieval model...")
        model = torch.load("./data/Pretrained_Model")                   # Load Model
        print("Loading corpus embeddings...")
        corpus_embeddings = torch.load("./data/Corpus_Embeddings.pt")   # Load Embeddings
        answer_ids = df[df['PostTypeId'] == 2]['Id'].tolist()
        validInput = True
        print("Done.")
    else:
        print("Invalid input. Input Y or N")
        userInput = input("First time processing? (Y/N): ")


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
        print("Exiting retrieval.")
        validInput = True
    else:
        print("Invalid input. Input Y or N")
        userInput = input("Input user query? (Y/N): ")
