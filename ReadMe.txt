As per the problem stated we have to predict the base score of the drugs.
So at first I checked all the data and removed any data with no base_score present.

Then next step I checked all abouut the review of the product. As the review matters for the product I had to check about how much it effects.
So I went towards NLP. And checked the sentiment of the review how positive, negative, neutral or compound it is.
I achieved this with nltk.sentiment.vader library in which there is a pre made sentiment analyzer which can categorize a sentence in above features.

I checked all the reviews and classified it into above 4 mentioned sentiments. After that I did same for testing data set.
For training I left out the patient_id and when the drug was approved as it doesn't effect the overall data as it doesn't matter what the values of this preset is.

After that I used label encoder to change the data of name and what the drug used for to numbers so that a value is associated with each type of drugs and disease.
After that I used Random Forest Regressor as model and trained the data in it for prediction.

