import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import decimal



def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.
    
    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    # Get word frequency for each file in each category
    word_spam_collection = util.get_word_freq(file_lists_by_category[0])
    word_ham_collection = util.get_word_freq(file_lists_by_category[1])

    # Create set to store all words in training files (i.e. master dictionary, W)
    all_words = set(word_spam_collection.keys()).union(set(word_ham_collection.keys()))

    # Numerical values required for p_d and q_d estimates
    D = len(all_words)
    sum_all_spam = sum(word_spam_collection.values())
    sum_all_ham = sum(word_ham_collection.values())

    # Create the dictionaries that are to be returned
    dictionary_for_p_d = {}
    dictionary_for_q_d = {}

    # Compute smoothed estimates of p_d and q_d for each word in the master dictionary
    for word in all_words:
        p_d = (word_spam_collection.get(word, 0) + 1) / (sum_all_spam + D)
        q_d = (word_ham_collection.get(word, 0) + 1) / (sum_all_ham + D)

        dictionary_for_p_d[word] = p_d
        dictionary_for_q_d[word] = q_d
    
    probabilities_by_category = (dictionary_for_p_d, dictionary_for_q_d)
    return probabilities_by_category


def classify_new_email(filename, probabilities_by_category, prior_by_category, list):
    """
    Use Naive Bayes classification to classify the email in the given file.
    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution
    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    # accumulating the information required to execute the MAP rule    
    x = util.get_word_freq([filename])    
    
    # calculating the likelihoods for X
    spam_likelihood = 0
    ham_likelihood = 0
    
    
    # calculating the likelihoods for X
    spam_likelihood = sum(np.log(probabilities_by_category[0].get(word, 1)) * count for word, count in x.items())
    ham_likelihood = sum(np.log(probabilities_by_category[1].get(word, 1)) * count for word, count in x.items())

    
    # calculating the MAP rule
    spam_or_ham = 'spam' if spam_likelihood - ham_likelihood > np.log(prior_by_category[1]) - np.log(prior_by_category[0]) + list else 'ham'


    # calculating the posterior probabilities for X
    posterior_of_x = np.exp(spam_likelihood + np.log(prior_by_category[0])) + np.exp(ham_likelihood + np.log(prior_by_category[1]))
    log_probability_posterior = [spam_likelihood + np.log(prior_by_category[0]) - np.log(posterior_of_x),ham_likelihood + np.log(prior_by_category[1]) - np.log(posterior_of_x)]
    
    #classify_result = (spam_or_ham, log_probability_posterior)
    
    return (spam_or_ham, log_probability_posterior)

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category,
                                                 0)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    error_list = []
    list = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    
    for i in list:
        new_performance_measures = np.zeros([2,2])
        
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category,
                                                     i)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename) 
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham') 
            new_performance_measures[int(true_index), int(guessed_index)] += 1 

        # All the values except the diagonal are the errors
        right_value = np.diag(new_performance_measures)
        right_sum = np.sum(new_performance_measures, 1)
        
        x = right_sum[0] - right_value[0]
        y = right_sum[1] - right_value[1]

        # append the error values to the list
        error_list.append((right_sum[0] - right_value[0], right_sum[1] - right_value[1]))
        
        # Plot the trade-off curve
        plt.plot(x,y, color = 'maroon', marker = 'x')

    # Trade-off curve specifications
    plt.xlabel('Type-1 Error')
    plt.ylabel('Type-2 Error')
    plt.title('Error Trade-off Curve')
        
    plt.savefig("nbc.pdf")
    plt.show()