# Presentation notes

## Opening

My project is called Movie Review Sentiment Analysis. It takes a movie review in English and predicts whether the opinion is positive or negative. It is an example of natural language processing and supervised learning.

## Dataset

I used the Cornell movie review dataset, version 2.0. It contains 2,000 reviews: 1,000 positive and 1,000 negative. I used 80% for training and kept 20% separate for testing.

## Method

The model cannot work directly with words, so I used TF-IDF to turn the text into numbers. It assigns weights to words and pairs of words. Logistic regression then learns which patterns are associated with positive and negative reviews. This connects to the course topics of machine learning, logistic regression and performance evaluation.

## Demo

First, I select the positive example and click Analyze review. Then I do the same with the negative example. Finally, I try a mixed review to show that the model still has to choose one of the two classes.

## Results

The model correctly classified 344 out of 400 test reviews, giving an accuracy of 86%. A classifier that always chooses one class would get 50% on this balanced test set. The model made 56 mistakes, so the result is useful but not perfect.

## Limits and next steps

The model can struggle with sarcasm, mixed opinions and short reviews. Its scores are estimates, not guarantees. A useful next step would be to compare it with a neural network and use cross-validation on training data before evaluating on the test set.

## Possible questions

**Why logistic regression?** It is small, fast and easier to explain. It also appears in the course syllabus.

**What does TF-IDF mean?** Term frequency-inverse document frequency. It gives more weight to terms that are frequent in one review but less common across the collection. This project uses a sublinear version of term frequency.

**Why separate training and test data?** To check whether the model works on examples it did not see while learning. TF-IDF is also fitted only on the training data.

**Is a 90% positive score the same as 90% accuracy?** No. The score is the model's estimate for one review. Accuracy is measured over labeled test examples.

**Is this a neural network?** No. It is a linear classifier with a sigmoid output for binary classification. A neural network could be a later comparison.

**Can it detect neutral reviews?** No. This dataset and model have only two classes.
