from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

import logging
logger = logging.getLogger(__name__)

class CreateEmbeddings:
    def __init__(self):
        self.embedding_model = None

    def _initialize_embedding_model(self):
        if self.embedding_model is None:
            model_name = "BAAI/bge-large-en-v1.5"
            encode_kwargs = {'normalize_embeddings': True}
            model_kwargs = {'device': 'mps'}
            self.embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                encode_kwargs=encode_kwargs,
                model_kwargs=model_kwargs
            )

    def create_embeddings(self,sentences):
        self._initialize_embedding_model()
        for x in tqdm(sentences,desc='Creating embeddings for semantic splitter'):
            embeddings = self.embedding_model.embed_documents(x['combined_sentence'])
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]
        return sentences

def combine_sentences(sentences, buffer_size=1):
    logger.info(f"combining sentences with window size {buffer_size}")
    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ''
        
        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']
        
        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined
                combined_sentence += ' ' + sentences[j]['sentence']
        
        #then add everything to your dict
        sentences[i]['combined_sentence'] = combined_sentence
    logger.info(f"Completed combining sentences. Number of sentences: {len(sentences)}")
    return sentences



def calculate_cosine_distances(sentences) :
    distances = []
    for i in tqdm(range(len(sentences) - 1),desc="computing cosine distances"):
        embedding_current = sentences [i] ['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        # Convert to cosine distance
        distance = 1 - similarity
        # Append cosine distance to the list
        distances.append(distance)
        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance
        # Optionally handle the last sentence
        # sentences|-1]['distance_to_next'] = None # or a default value
    return distances, sentences


def semantic_splitting(distances,sentences,percentile_threshold=None,sigma_multiplier_threshold=None,IQR_multiplier_threshold=None):
    #get distance threshold for where to split 
    if percentile_threshold:
        distance_threshold = np.percentile(distances,percentile_threshold)
    elif sigma_multiplier_threshold:
        distance_threshold = sigma_multiplier_threshold * np.std(distances)
    elif IQR_multiplier_threshold:
        sorted_distance = np.sort(distances)
        Q1 = np.percentile(sorted_distance, 25, interpolation = 'midpoint') 
        Q2 = np.percentile(sorted_distance, 50, interpolation = 'midpoint') 
        Q3 = np.percentile(sorted_distance, 75, interpolation = 'midpoint')
        IQR = Q3 - Q1
        distance_threshold = Q3 + IQR_multiplier_threshold * IQR
    else:
        logger.info("splitting method not passed. using default 95% percentile method")
        distance_threshold = np.percentile(distances,0.95)
    
    #how many break points above the threshold
    num_breaks = len([x for x in distances if x > distance_threshold])
    
    #indices of distances above the threshold
    indices_above_thresh = [i for i,x in enumerate(distances) if x > distance_threshold]

    start_index = 0
    chunks = []
    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is one less than the current breakpoint
        end_index = index - 1
        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index

    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
    return chunks