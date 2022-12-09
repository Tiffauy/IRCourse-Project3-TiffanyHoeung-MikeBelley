# Overview
The RPG Stack Exchange website offers a place for game masters and players of tabletop role-playing games to ask and answer questions. It is a great 
resource for those that enjoy tabletop RPGs and the website supports questions about many different RPG systems such as Dungeons \& Dragons, World of Darkness, 
Pathfinder and Apocalypse World. We decided to use this website for our search system because we are both familiar with the many aspects of tabletop gaming 
and it’s generally a topic we enjoy reading about. The RPG world is filled with many different types of players, from players that enjoy more roleplay-based 
games, to players that focus on minimizing and maximizing their character’s skills and attributes. Regardless of playstyle, the RPG Stack Exchange offers 
many interesting and engaging discussions for all types of players to read and enjoy.

For this project, we used the HuggingFace API to use and modify a bi-encoder retrieval model. 

# Usage
The large file consisting of the posts of the RPG Stack Exchange website is not included. You have to manually download it and drop the Posts.xml file into the data folder. Once you have done this, the two PreTrained and FineTuned files should run independently. However, to save time, included in the data folder are the PreTrained and FineTuned models that can be loaded into the workspace.

The two different retrieval systems explored in this project are:
* all-MiniLM-L6-v2 PreTrained
* all-MiniLM-L6-v2 FineTuned

# all-MiniLM-L6-v2 PreTrained
The MiniLM-L6 model, more specifically all-MiniLM-L6-v2, is a sentence-transformers model. According to the HuggingFace documentation, this model maps sentences and paragraphs to a 384 dimensional vector space to be used for different tasks; in the case of this project, the task applied to the model was semantic search.

To use this on our corpus, we first loaded the MiniLM-L6 model into our workspace. We then passed in a list of texts to the model, where each entry was an answer to a question from the RPG Stack Exchange site. The model, given this list, created embeddings for each body of text. This vector representation is meant to capture the semantic meaning of the information. Likewise, the queries were then also encoded into their embedding representations. With the list of query and answer embeddings, the model is able to perform a semantic search query by query and return the top 1000 results for each. 

# all-MiniLM-L6-v2 FineTuned
The second model we explored for this project takes the previous model, the all-MiniLM-L6-v2 model, and trains it across the RPG Stack Exchange corpus. Our parameters for finetuning the model is as follows:
* Epochs: 45
* Batch Size: 64
* Max Sequence Length: 256

We selected the MultipleNegativesRankingLoss as our loss function.
