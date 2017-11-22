# Deep-Learning-for-Knowledge-Graph-Completion


Below is a list of the contents of the repository. Please update when you commit.
###  data/ Contains all the data we've built for this project. 
	<b>EncodedData.csv:</b>  The Main data, already encoded with entity embeddings, and 
					  associated outputs. Of the form 
					  [EntityA|EntityB|a0,a1,...,a299|b0,b1,...,b299|p0,p1,p2,p3,p4]
					  where EntityA/B are the names of the entities, ai/bi are the 
					  embeddings for these entities, and pi is the predicate for this
					  entity-entity pair. Predicates follow {1: True, 0: Unknown, -1: False}.
	Build_Data.py:	  The Python script used to generate EncodedData.csv
	Build_Data.ipynb: A Jupyter Notebook for the Build_Data.py script

###		data/raw/		  	A directory containing the raw exported knowledge graph
		SparqlQuery.txt:		The query used to pull the data from WikiData
		wikiData.txt:			The result of the above query on WikiData
		wikiData.csv:			Same as wikiData.txt but in .csv format
		wikiData.py:			A python script used to generate much of processed/

### 	data/embeddings/	A directory of word/entity embeddings and their scripts
		entity_embeddings.pkl:  A dictionary of entity embeddings
		word_vectors.pkl:		A dictionary of word embeddings
		word_embeddings.py:		Script for generating word_vectors.pkl

###		data/processed/	  	A directory of processed versions of the data in raw/
		entities_map.pkl:		A dictionary that maps entity labels to their full names
		entities.txt:			A list of all the entities' full names
		positiveTriplets: 		The list of true statements for entity-entity pairs
		negativeTriplets:		A list of some false statements for entity-entity pairs
		predicates.txt: 		A list of the 5 predicates used for this project
		unique.txt:				A list of the unique elements of vocab.txt
		vocab.txt:				All words used throughout the entities.txt file



