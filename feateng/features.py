# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log, sqrt
from numpy import mean
import gzip
import json
import spacy
import nltk
import difflib
import nltk
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """


        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?

        guess_length = 0
        guess_length = log(1 + len(guess))

        # How many words long is the question?


        # How many characters long is the guess?
        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", guess_length)  

class FrequencyFeature(Feature):
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:                                                
            self.counts[self.normalize(ii["page"])] += 1                    

    def __call__(self, question, run, guess, guess_history, other_guesses=None):                
        yield ("guess", log(1 + self.counts[self.normalize(guess)])) 

class FrequencyNormalizedFeature(Feature):
    """
    This feature modifies the frequency feature by taking the z-score.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:                                                
            self.counts[self.normalize(ii["page"])] += 1  

        # Calculate mean and standard deviation of counts
        all_counts = list(self.counts.values())
        self.mean = sum(all_counts) / len(all_counts)  # Mean
        variance = sum((x - self.mean) ** 2 for x in all_counts) / len(all_counts)
        self.std = sqrt(variance)  # Standard deviation                  

    def __call__(self, question, run, guess, guess_history, other_guesses=None): 
        guess_count = self.counts[self.normalize(guess)]
        
        # Z-score normalization (handle division by zero if std is zero)
        if self.std != 0:
            z_score = (guess_count - self.mean) / self.std
        else:
            z_score = log(1 + self.counts[self.normalize(guess)]) # Return log normalized count if std is zero
        
        yield ("guess", z_score)

class LengthPlusFrequencyNormalizedFeature(Feature):
    """
    This feature modifies the normalized frequency feature by adding length feature.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:                                                
            self.counts[self.normalize(ii["page"])] += 1

        # Calculate mean and standard deviation of counts
        all_counts = list(self.counts.values())
        self.mean = sum(all_counts) / len(all_counts)  # Mean
        variance = sum((x - self.mean) ** 2 for x in all_counts) / len(all_counts)
        self.std = sqrt(variance)  # Standard deviation       

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        guess_count = self.counts[self.normalize(guess)]
        
        # Z-score normalization (handle division by zero if std is zero)
        if self.std != 0:
            z_score = (guess_count - self.mean) / self.std
        else:
            z_score = log(1 + self.counts[self.normalize(guess)]) # Return log normalized count if std is zero
        
        yield ("guess", z_score + log(1 + len(guess)))  # Return z_score for frequency with length of the guess   

class KeywordPresenceFeature(Feature):
    """
    This feature checks the presence of keywords in the guess with run part of the question.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.keywords = set()
        self.normalize = normalize_answer
    
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # Normalize the run and guess
        normalized_run = self.normalize(run) if run else ""
        normalized_guess = self.normalize(guess) if guess else ""

        keywords = normalized_guess.split(" ")

        # Check if any of the keywords are present in the question
        question_keyword_presence = any(keyword in normalized_run for keyword in keywords)

        # Return features
        yield ("guess", 1 if question_keyword_presence else 0)

class KeywordOverlapFeature(Feature):
    """
    This feature checks the number of overlaps of keywords in the guess with run part of the question.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.keywords = set()
        self.normalize = normalize_answer

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # Normalize the question and guess
        normalized_run = self.normalize(run) if run else ""
        normalized_guess = self.normalize(guess) if guess else ""
        
        keywords = normalized_guess.split(" ")

        # Check if any of the keywords are present in the question
        keyword_count = 0
        for keyword in keywords:
            if keyword in normalized_run:
                keyword_count += normalized_run.count(keyword)

        # Return features
        yield ("guess", log(1 + keyword_count))

class KeywordOverlapDistributionFeature(Feature):
    """
    This feature checks the number of overlaps of keywords in the guess with question and give the frequency using a training set.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.keywords = set()
        self.counts = Counter()
        self.normalize = normalize_answer
    
    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:                              
            # Normalize the run and guess
            normalized_text = self.normalize(ii["text"]) if ii["text"] else ""
            normalized_page = self.normalize(ii["page"]) if ii["page"] else ""

            keywords = normalized_page.split(" ")

            # Check if any of the keywords are present in the question
            keyword_overlap_count = 0
            for keyword in keywords:
                if keyword in normalized_text:
                    keyword_overlap_count += normalized_text.count(keyword)
            
            self.counts[keyword] += keyword_overlap_count      

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # Normalize the question and guess
        normalized_run = self.normalize(run) if run else ""
        normalized_guess = self.normalize(guess) if guess else ""
        
        keywords = normalized_guess.split(" ")

        # Check if any of the keywords are present in the question
        keyword_overlap_counts = 0
        for keyword in keywords:
            if keyword in normalized_run:
                keyword_overlap_counts += self.counts[self.normalize(keyword)]
                if keyword_overlap_counts == 0:
                    keyword_overlap_counts += normalized_run.count(keyword)
        
        # Return features
        yield ("guess", log(1 + keyword_overlap_counts))

class NamedEntitiesFeature(Feature):
    """
    Check the presence of named entities in the guess and output the count.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer                  

    def __call__(self, question, run, guess, guess_history, other_guesses=None):  
        doc = nlp(guess)   
        yield ("guess", log(1 + len(doc.ents)))

class NamedEntitiesNormalizedFeature(Feature):
    """
    Check the presence of named entities in the normalized guess and output the count.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer                  

    def __call__(self, question, run, guess, guess_history, other_guesses=None):  
        doc = nlp(self.normalize(guess)) 
        yield ("guess", log(1 + len(doc.ents)))

class PartialNamedEntitiesFeature(Feature):
    """
    Check the presence of specified named entities in the normalized guess and output the count.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer                  
    def __call__(self, question, run, guess, guess_history, other_guesses=None):  
        doc = nlp(self.normalize(guess))
        entities_from_text = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "NORP")]

        yield ("guess", log(1 + len(entities_from_text)))

class KeywordPresencePlusNamedEntitiesFeature(Feature):
    """
    This feature combines KeywordPresence and NamedEntitiesFeature.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.keywords = set()
        self.normalize = normalize_answer

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # Normalize the question and guess to lower case for comparison
        normalized_run = self.normalize(run) if run else ""
        normalized_guess = self.normalize(guess) if guess else ""
        
        keywords = normalized_guess.split(" ")

        # Check if any of the keywords are present in the question
        question_keyword_presence = any(keyword in normalized_run for keyword in keywords)

        doc = nlp(guess)  
        if len(doc.ents) == 0 and question_keyword_presence:  
            yield ("guess", 0)  
        elif len(doc.ents) != 0 and question_keyword_presence:  
            yield ("guess", 1)    
        else:                           
            yield ("guess", 0)

class SynonymFeature(Feature):
    """
    This feature check the synonyms of the guess with run and outputs the count.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer 
    # Function to find synonyms using WordNet
    def get_synonyms(self, word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms                 
    def __call__(self, question, run, guess, guess_history, other_guesses=None): 
        # Tokenize both guess and answer into words
        guess_words = word_tokenize(self.normalize(guess))
        run_words = set(word_tokenize(self.normalize(run))) 

        # For each word in the answer, check if any of its synonyms appear in the guess
        synonym_word_count = 0
        for guess_word in guess_words:
            synonyms = self.get_synonyms(guess_word)
            # Check if any synonym is present in the guess
            if run_words.intersection(synonyms):
                synonym_word_count += 1
        
        yield ("guess", log(1 + synonym_word_count))        

class DistanceFeature(Feature):
    """
    This feature checks word difference between guess and a part of the run.
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer            
    def __call__(self, question, run, guess, guess_history, other_guesses=None): 
        # Tokenize both guess and answer into words
        guess_words = word_tokenize(self.normalize(guess))
        run_words = word_tokenize(self.normalize(run))

        # Calculate similarity ratio using difflib
        distance = 0
        for guess_word in guess_words:
            for i, run_word in enumerate(run_words):
                similarity = difflib.SequenceMatcher(None, guess_word, run_word).ratio()
                distance += similarity 
                if i >= 5:
                    break

        # print(distance)
        yield ("guess", log(1 + distance))       

if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from parameters import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags, guesser_params)
    buzzer = load_buzzer(flags, buzzer_params)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
