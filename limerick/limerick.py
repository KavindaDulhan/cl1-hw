# Author: Kavinda Kehelella
# Date: 08/30/2024

# Use word_tokenize to split raw text into words
import nltk
import json

from nltk.tokenize import word_tokenize
from string import punctuation, digits
import re
nltk.download('punkt_tab')

class LimerickDetector:
    @staticmethod
    def load_json(filename):
        with open("sample_limericks.json") as infile:
            data = json.load(infile)

        limericks = []
        for example in data:
            limericks.append((example["limerick"], "\n".join(example["lines"])))

        return limericks
        
    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()
        self._vowels = lambda x: [y for y in x if y[-1] in digits]

    def _normalize(self, a):
        """
        Do reasonable normalization so we can still look up words in the
        dictionary
        """

        return a.lower().strip()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.  
        """

        # TODO: Complete this function

        word = self._normalize(word)
        if word not in self._pronunciations:
            return 1  # If there is no entry in the dictionary, return 1
    
        # Get all the pronunciations and then put syllables to a list
        syllable_counts = [len(self._vowels(pron)) for pron in self._pronunciations[word]]
        return min(syllable_counts)  # Return the minimum syllable count
    
    def after_stressed(self, word):
        """
        For each of the prounciations, yield whatever is after the
        last stressed syllable.  If there are no stressed syllables,
        return the whole proununciation.
        """

        # TODO: Complete this function

        pronunciations = self._pronunciations.get(self._normalize(word), [])
        
        for pronunciation in pronunciations:
            stressed_idx = -1

            # Find the index of the last stressed vowel
            for i, phoneme in enumerate(pronunciation):
                if phoneme[-1] in '12':  # Primary (1) or secondary (2) stress
                    stressed_idx = i
            
            if stressed_idx != -1:
                yield pronunciation[stressed_idx:]  # Yield everything after the last stressed vowel including the vowel according to piazza response
    
    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.

        We use the definition from Wikipedia:

        Given two pronuncation lookups, see if they rhyme.  We use the definition from Wikipedia:

        A rhyme is a repetition of the exact phonemes in the final
        stressed syllables and any following syllables of words.

        """
        # TODO: Complete this function
        # Look up the pronunciations and get the prounciation after
        # the stressed vowel
        a = self._normalize(a)
        b = self._normalize(b)
        
        # Extract everything after the last stressed vowel including the vowel in each pronounciation
        pronunciations_a = list(self.after_stressed(a))
        pronunciations_b = list(self.after_stressed(b))
        
        # Check for rhymes
        for pron_a in pronunciations_a:
            for pron_b in pronunciations_b:
                if pron_a == pron_b:
                    return True

        return False

    def last_words(self, lines):
        """
        Given a list of lines in a list, return the last word in each line
        """
        # TODO: Complete this function

        words = []
        for line in lines:
            tokens_puncts = word_tokenize(line)

            # Filter out punctuation from the tokens
            tokens = [token for token in tokens_puncts if token not in punctuation]

            words.append(tokens[-1] if tokens else "")
        return words

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and possibly the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """

        text = text.strip()
        lines = text.split('\n')

        # TODO: Complete this function
        
        if len(lines) != 5:
            return False  # A limerick must have exactly 5 lines
        
        last_words = self.last_words(lines)
        
        # Check for limerick AABBA
        if self.rhymes(last_words[0], last_words[1]) and self.rhymes(last_words[0], last_words[4]) and self.rhymes(last_words[2], last_words[3]) and \
        not self.rhymes(last_words[0], last_words[2]):
            return True

        return False
    
    # Extra credit functions
    def apostrophe_tokenize(self, text):
        """
        This function handles apostrophes in words correctly so that "can’t" would rhyme with "pant" as an example.
        """
        return re.findall(r"\b\w+'\w+|\w+\b", text)
    
    def guess_syllables(self, word):
        """
        This function guesses about the number of syllables in unknown words not found in the dictionary.
        
        Here, this counts number of groups of vowels in the word and return the count. If there are no vowels then return 1.

        As an example "aehdsiopypapa" returns 3 since it finds the list of vowel groups as ['ae', 'io', 'y', 'a', 'a']
        """
        word = self._normalize(word)
        return len(re.findall(r'[aeiouy]+', word))


if __name__ == "__main__":
    ld = LimerickDetector()

    limerick_tests = ld.load_json("sample_limericks.json")
    
    words = ["billow", "pillow", "top", "America", "doghouse", "two words", "Laptop", "asdfasd"]

    for display, func in [["Syllables", ld.num_syllables],
                          ["After Stressed", lambda x: list(ld.after_stressed(x))],
                          ["Rhymes", lambda x: "\t".join("%10s%6s" % (y, ld.rhymes(x, y)) for y in words)]]:
        print("=========\n".join(['', "%s\n" % display, '']))
        for word in words:
            print("%15s\t%s" % (word, str(func(word))))

    print(limerick_tests)
    for result, limerick in limerick_tests:
        print("=========\n")
        print(limerick)
        print("Truth: %s\tResult: %s" % (result, ld.is_limerick(limerick)))
