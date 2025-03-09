import ssl
import certifi
import re
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import reduce

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
inputFile = open("SMSSpamCollection.txt", "r")
spamContent = inputFile.read().splitlines() 







'''
Main function
    -train the model using the training data
    -make predictions on the testing data and test data
    -compute and log performance metrics (results.log or results.txt)'''

def cleanText(content):
    '''
    This function takes in a file's content, makes all text into lowercase, removes all special characters, and returns a list of lines.
    '''
    content_lines = []
    stop_words = set(stopwords.words('english'))     #Create a set of all stop words
    for line in content:                        #Go through each line in the content
        line = line.lower()                         #Make all text lowercase
        line = re.sub(r'[^a-zA-Z0-9\s]', ' ', line)   #Remove all special characters  found online https://www.geeksforgeeks.org/python-remove-all-characters-except-letters-and-numbers/
        #print("check 1:",type(line))
        word_tokens = word_tokenize(line)            #Tokenize the line
        #line = [w for w in word_tokens if not w.lower() in stop_words] #Remove all stop words
        temp_words = []                         
        for w in word_tokens:                         #Remove all stop words
            if not w.lower() in stop_words:
                temp_words.append(w)
        line = temp_words
        line = ' '.join(line)  
        temp_words = []
        #print("check 2:",type(line))
        word_tokens = word_tokenize(line)
        line = reduce(lambda x, y: x + " " + PorterStemmer().stem(y), word_tokens, "")
        #line = ' '.join(line)                       #Join the list of words back into a string
        #print(type(line))
        content_lines.append(line)                  #Add the cleaned line to the list
    return content_lines                            #Return the list of cleaned lines

def seperateSpamAndHam(content):
    '''
    This function takes in a list of lines and returns two lists, one for spam and one for ham.
    '''
    spamContent = []
    hamContent = []
    for line in content:                    # Go through each line in the content 
        if line.startswith("spam"):             # If the line starts with spam, add it to the spam list
            parts = line.split(' ', 1)              # Split the line at the first space to remove the spam label
            if len(parts) > 1:                      # Ensure there are at least two parts after splitting
                tempLine = parts[1]                     # Create a string without "spam"
            else:
                tempLine = parts[0]                     # Use the whole line if no space is found
            spamContent.append(tempLine)            # Add the string to the spam list
        else:                                   # If the line does not start with spam, add it to the ham list
            parts = line.split(' ', 1)              # Split the line at the first space to remove the ham label
            if len(parts) > 1:                      # Ensure there are at least two parts after splitting
                tempLine = parts[1]                     # Create a string without "ham"
            else:
                tempLine = parts[0]                     # Use the whole line if no space is found
            hamContent.append(tempLine)             # Add the string to the ham list
    return spamContent, hamContent                  # Return the spam and ham lists

def createSpamAndHamDict(spamContent, hamContent):
    spamDict = {}                           #Create a dictionary for all the words in spam, use each word as a key and each value is the number of times a word appears
    hamDict = {}                            #Create a dictionary for all the words in ham, use each word as a key and each value is the number of times a word appears

    for line in spamContent:                #Go through each line in spamContent
        words = line.split()                    #Split each line into individual words
        for word in words:                          #Go through each word in the line
            if word in spamDict:                        #If the word is in the dictionary, increase the value at that key by 1
                spamDict[word] += 1                         #Increase the word amount by 1
            else:                                       #If the word is not in the dictionary, add it and set the valu to 1
                spamDict[word] = 1                          #Add a new word to the dictionary and set to 1

    for line in hamContent:                 #Repeat the same process as done for Spam
        words = line.split()
        for word in words:
            if word in hamDict:
                hamDict[word] += 1
            else:
                hamDict[word] = 1
    
    return spamDict, hamDict                #Return a dictionary of spam words and anoter of ham words

contentLines = cleanText(spamContent)       #Clean the text

for i in contentLines:
#    print(type(i))
    print(i)

spamLines, hamLines = seperateSpamAndHam(contentLines)  #Seperate spam and ham into two lists

#for i in spamLines:
#    print(i)

spamDict,hamDict = createSpamAndHamDict(spamLines,hamLines)     #Create dictionaries for each word set
