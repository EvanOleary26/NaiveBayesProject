import ssl
import certifi
import re
import nltk
import random
import time
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')

class DataPreprocessing:
    
    def __init__(self):
        lines = []
        
    
    def preprocess(self,inputFile):
        '''
        preprocess(self, inputFile):
        - take in a list of lines from a file
        - convert all text to lowercase
        - remove all special characters
        - remove all stop words
        - rempove all special characters
        - apply stemming
        - return a list of lines
        '''
        
        content_lines = []
        stop_words = set(stopwords.words('english'))  # Create a set of all stop words
        
        for line in inputFile:                        # Go through each line in the content
            line = line.lower()                           # Make all text lowercase
            line = re.sub(r'[^a-zA-Z0-9\s]', ' ', line)   # Remove all special characters found online https://www.geeksforgeeks.org/python-remove-all-characters-except-letters-and-numbers/
            
            word_tokens = word_tokenize(line)            # Tokenize the line
            temp_words = []                              # Create temp words
            for w in word_tokens:                           # Go through all words in the line
                if not w in stop_words:                         # Check if the words are not stop words
                    temp_words.append(w)                        # If they arent stop words add them to the temp_words list
            line = temp_words                            # Set line to temp_words
            line = ' '.join(line)                        # Connect the words into one single string
            temp_words = []                              # Reset temp_words to empty
            word_tokens = word_tokenize(line)            # Reset word_tokens to be a line, but without stop words
            line = reduce(lambda x, y: x + " " + PorterStemmer().stem(y), word_tokens, "")  # Get the stem of the words in a line, and combine them into a strin
            line = line.replace(' ','',1)
            content_lines.append(line)                   # Add the cleaned line to the list content_lines
        return content_lines
            
            
    def load_data(self):
        '''
        load_data(self):
        - read in the file
        - seperate labels from text
            - spam = 1, ham = 0
        - return a list of lists: 
            - index 0 = 0 or 1 for spam or ham
            - index 1 = the string of the rest of the message
        '''
        
        inputFile = open("SMSSpamCollection.txt", "r")      #Open the "SMSSpamCollection.txt" file
        fileInputLines = inputFile.read().splitlines()      #Split the inputFile into a list with each line of the text
        
        
        content_lines = self.preprocess(fileInputLines)     #Preprocess the data from the file, load it into content_lines
        labeledLines = []
        for line in content_lines:                    # Go through each line in the content
            if line.startswith("spam"):             # If the line starts with spam, add it to the spam list
                line = line.replace("spam ","")              # Split the line at the first space to remove the spam label
                labeledLines.append([1,line])            # Add the string with the spam identifier
            else:                                   # If the line does not start with spam, add it to the ham list
                line = line.replace("ham ","")              # Split the line at the first space to remove the ham label
                labeledLines.append([0,line])             # Add the string with the ham identifier

        inputFile.close()
        return labeledLines                         # Return a list with each element being a list wiht the binary label and the text of the message
    
    def split_data(self,content_lines):
        '''
        split_data():
        - input: a list of lists containing the identifier for all messages and their content
        - shuffle the data
        - split the data into training and testing sets (80% - 20% split)
        - return two lists
        '''
        seed = int(time.time()) % 1000
        random.seed(seed)
        random.shuffle(content_lines)
        
        split_index = int(0.8 * len(content_lines))  # Calculate the split index for 80-20 split

        train_data = content_lines[:split_index]  # First 80% for training
        test_data = content_lines[split_index:]   # Remaining 20% for testing

        return train_data, test_data  # Return the training and testing sets
            