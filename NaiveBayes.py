
class NBClassifier:
    def __init__(self):                                 #Default constructor  
        self.class_priors = {}                              #Class Priors for storing the percent chance of each word being in a message
        self.word_counts = {'spam': {}, 'ham': {}}          #Used to get the amount of each word and if it is spam or ham
        self.total_counts = {'spam': 0, 'ham': 0}           #Used to count the total number of spam or ham words
        self.vocab = set()                                  #Create vocab set so that only one of every word is included

    def train(self, train_data):
        '''
        train(self, train_data):
        - compute class priors : p(ham) and p(spam)
        - count word occurances separately for spam and ham
        - compute probabilities of words given a class p(word|ham), p(word|spam) with Laplace smoothing
        '''
        spam_count = 0
        ham_count = 0
        for label,message in train_data:
            if label == 1:                                      #Check if the word message is a spam message
                spam_count += 1                                 #Increase the count of spam messages by 1
                splitMessage = message.split()                  #Seperate a message (String) into a list of strings 
                for word in splitMessage:                       #Go through every word in spam messages
                    self.vocab.add(word)                        #Add word to vocab list
                    self.total_counts['spam'] += 1              #Count the amount of words in spam
                    if word in self.word_counts['spam']:        #Check if word is already in the spam dictionary
                        self.word_counts['spam'][word] += 1         #If it is increase the count for that word by 1
                    else:
                        self.word_counts['spam'][word] = 1          #If not create a key for the word
            else:                                               #If the message is not spam then it is ham            
                ham_count += 1                                  #Increase the count of ham messages by 1
                splitMessage = message.split()                  #Seperate a message (String) into a list of strings
                for word in splitMessage:                       #Go through every word in ham messages
                    self.vocab.add(word)                        #Add word to vocab list
                    self.total_counts['ham'] += 1               #Count the amount of words in ham
                    if word in self.word_counts['ham']:         #Check if word is already in the ham dictionary
                        self.word_counts['ham'][word] += 1          #If it is increase the count for that word by 1
                    else:
                        self.word_counts['ham'][word] = 1           #If not create a key for the word
        total_messages = spam_count+ ham_count                  #Create a variable for the total amount of messages
        self.class_priors['spam'] = spam_count / total_messages #Set Spam priors to the percent of messages that are spam
        self.class_priors['ham'] = ham_count / total_messages   #Set Ham priors to the percent of messages that are ham 
        
        self.word_probs = {'spam': {}, 'ham': {}}               #Create a dictionary to hold all words probability
        vocab_size = len(self.vocab)                            #Create variable to hold the length of vocab list

        for word in self.vocab:
            self.word_probs['spam'][word] = (self.word_counts['spam'].get(word,0) + 1) / (self.total_counts['spam'] + vocab_size)
            self.word_probs['ham'][word] = (self.word_counts['ham'].get(word,0) + 1) / (self.total_counts['ham'] + vocab_size)

    def prediction(self, testing_data):
        '''
        prediction():
            - compute the probability of a message being spam or ham
            - assign the label with the higher probability
        '''
        predictions = []

        for message in testing_data:
            # Initialize log probabilities for spam and ham
            spam_prob = self.class_priors['spam']
            ham_prob = self.class_priors['ham']

            # Split the message into words
            words = message.split()

            # Compute the log probabilities for each word in the message
            for word in words:
                if word in self.vocab:  # Only consider words in the vocabulary
                    spam_prob *= self.word_probs['spam'].get(word, 1 / (self.total_counts['spam'] + len(self.vocab)))
                    ham_prob *= self.word_probs['ham'].get(word, 1 / (self.total_counts['ham'] + len(self.vocab)))

                # Assign the label based on the higher probability
            if spam_prob > ham_prob:
                predictions.append(1)  # Spam
            else:
                predictions.append(0)  # Ham

        return predictions