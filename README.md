# Machine-Learning-NaiveBayes
Implementation of Naive Bayes algorithm for Spam detection
Developed by Sean Taylor Thomas 
Oct. 2021

"spam-text.txt" is an input text file with the following qualifications:
  - each line begins with "spam" or "ham", the former indicating the rest of the line contains an example of spam text/email, the latter indicating otherwise
  - each line in the file represents an example for the algorithm to take into account

main.py takes in "spam-text.txt" and utilizes the examples contained in the file to develop an algorithm to decide whether new input is spam or not.
  - this algorithm is based on naive bayes theorem, which calculates the probability that a given new (non-training) example is
  1) spam,
  2) non-spam
  - after calculating the probability of each, the higher probability is simply the prediction by the algorithm.

** Program **
1) the program begins by splitting the data contained in "spam-text.txt" into training and testing datasets, then performs its calculations and outputs the accuracy obtained by the test data
2) the program then asks the user if he/she would  like to input his/her own email/sms example to have the program predict
3) the program repeats this process, outputting the prediction after each user input until "no" is answered

Other files:
- Naive-bayes-diabetes.py is the implementation of naive bayes for the dataset "pima-indians-diabetes.csv"
