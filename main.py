# ------------------------------------------------------------------
# Sean Taylor Thomas
# Implementing Naive Bayes Algorithm for Spam detection from scratch
# October 2021
#
# -----------------------------------------------------------------

def load_file(filename):
    dataset = []
    with open(filename) as f:
        dataset = f.readlines()
    return dataset


def split_data(dataset):
    spam = []
    non_spam = []
    for example in dataset:
        if example[0:3] == "spa":
            spam.append(example[5:-1].lower())
        elif example[0:3] == "ham":
            non_spam.append(example[4:-1].lower())
        else:
            print("No classification!")
    return [spam, non_spam]


def build_vocabulary(dataset):
    """ Create a list of all unique words in class"""
    vocab = []
    for message in dataset:
        words = message.split()
        for word in words:
            if word not in vocab:
                vocab.append(word)
    return vocab


def build_frequency(dataset, vocab):
    """ create list of frequencies for each unique word in class"""
    freq = [0.0] * len(vocab)
    for message in dataset:
        words = message.split()
        for word in words:
            if word in vocab:
                freq[vocab.index(word)] += 1.0
    # divide every element by total words of class
    freq1 = []
    for elem in freq:
        freq1.append(elem / len(freq))
    return freq1


def sep_data(dataset):
    """ Separating data into training and testing lists"""
    test = []
    train = []
    for x in range(len(dataset)):
        if x / len(dataset) <= .85:
            train.append(dataset[x])
        else:
            test.append(dataset[x])
    return [train, test]


def naive_bayes(test, spam_dict, n_spam_dict, prob_spam, prob_n_spam):
    """ calculating probabilities that a single test """
    probability_SPAM = prob_spam
    probability_NOT_SPAM = prob_n_spam
    test_words = test.split()
    for word in test_words:
        # calculating P(x | SPAM)
        if spam_dict.get(word) is not None:
            probability_SPAM *= spam_dict[word]
        else:  # test word not in spam dict, utilize laplace
            probability_SPAM *= 1 / (2 * len(spam_dict))
        # calculating P(x | non-SPAM)
        if n_spam_dict.get(word) is not None:
            probability_NOT_SPAM *= n_spam_dict[word]
        else:  # test word not in n-spam dict, utilize laplace
            probability_NOT_SPAM *= 1 / (2 * len(n_spam_dict))
    return [probability_SPAM, probability_NOT_SPAM]


def test_function(test_data):
    """ Iterate through test_data and calculate accuracy, returns output"""
    output = [0, 0, 0, 0]  # counts of [true spam, false spam, true non-spam, false non-spam]
    split_test_data = split_data(test_data)
    is_spam_set = 2
    for classes in split_test_data:
        is_spam_set -= 1
        for single_test_data in classes:
            output_bayes = naive_bayes(single_test_data, spam_dict, non_spam_dict, prob_spam, prob_non_spam)
            probability_spam_test = output_bayes[0]
            probability_not_spam_test = output_bayes[1]
            if probability_spam_test > probability_not_spam_test and is_spam_set == 1:
                # print("We think it's spam, and we're correct")
                output[0] += 1
            elif probability_spam_test > probability_not_spam_test and is_spam_set != 1:
                # print("We think it's spam, but we're incorrect")
                output[1] += 1
            elif probability_spam_test < probability_not_spam_test and is_spam_set == 1:
                # print("We think it's not spam, but we're incorrect")
                output[2] += 1
            elif probability_spam_test < probability_not_spam_test and is_spam_set != 1:
                # print("We think it's not spam, and we're correct")
                output[3] += 1
            else:
                print("They are equal")
        return output


# Load data and separate it into training and testing sets
dataset = load_file('spam-data.txt')
separate_data = sep_data(dataset)
training_data = separate_data[0]
test_data = separate_data[1]

# Organize training data into two classes: spam, and not spam
training_data = split_data(training_data)
# create lists for spam and non-spam classes:
spam = training_data[0]
non_spam = training_data[1]

# Calculate probabilities P(spam) and P(not_spam)
prob_spam = len(spam) / (len(spam) + len(non_spam))
prob_non_spam = len(non_spam) / (len(spam) + len(non_spam))

# build vocabularies for each class, spam and not spam
spam_vocab = build_vocabulary(spam)
non_spam_vocab = build_vocabulary(non_spam)

# build corresponding list to calculate relative frequency of each word in each class
spam_freq = build_frequency(spam, spam_vocab)
non_spam_freq = build_frequency(non_spam, non_spam_vocab)

# create dictionaries for spam & non-spam classes where | key= word | value= frequency of word in class
spam_dict = dict(zip(spam_vocab, spam_freq))
non_spam_dict = dict(zip(non_spam_vocab, non_spam_freq))

# run test data through and calculate accuracy
test_metrics = test_function(test_data)
accuracy = (test_metrics[0] + test_metrics[3]) / (sum(test_metrics)) * 100
precision = test_metrics[0] / (test_metrics[0] + test_metrics[1] )
recall = test_metrics[0] / (test_metrics[0] + test_metrics[2])
f1_score = (2 * (recall * precision))/(precision + recall)
print("Accuracy of test data from 'spam-data.txt': %.2f%%  " % accuracy)
print("Precision of test data from 'spam-data.txt': %.2f%%  " % (precision*100))
print("Recall of test data from 'spam-data.txt': %.2f%%  " % (recall*100))
print("F1_Score of test data from 'spam-data.txt': %.2f  " % f1_score)



user_input = "yes"
while user_input.lower() != "no" and user_input.lower() != "n" and user_input.lower() != "q":
    # Ask user
    user_input = input("Would you like to enter a single SMS or email for the program to predict? (y/n): ")
    single_input = ""
    if user_input.lower() == "yes" or user_input.lower() == "y":
        single_input = input("Enter SMS/email to be predicted: ")
        single_input_prediction = naive_bayes(single_input, spam_dict, non_spam_dict, prob_spam, prob_non_spam)
        if single_input_prediction[0] > single_input_prediction[1]:
            print("Your message was predicted to be spam.")
        else:
            print("Your message was predicted NOT to be spam.")


