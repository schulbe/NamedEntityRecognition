# Named Entity Recognition (NER) on noisy data set

Named Entity Recognition (NER) describes the task of finding a certain reoccuring entities in texts.
Examples would be Names, Brands, Birthdays, Cities, etc. to help automatic text comprehension.

## About this Project
Searching for pretrained NER taggers, one will find numerous 
advanced approaches already integrated in well known NLP packages like *spacy*. 
Most of these achieve highest scores on comparative datasets, if it comes to applying them to practical
problems however, performance decreases significantly as soon as grammar and spelling are not 100% correct anymore
(as it realistically is the case in todays online style of writing).
This is especially true for german language as can be seen when analyzing named entities of the german example sentence

`Ich habe gestern Sabine Müller getroffen.` (I met Sabine Müller yesterday.)

In this context `Sabine Müller` gets recognized as a `PERSON` by spacy. However **None** of these will:

`Ich habe gestern sabine Müller getroffen.`

`Ich habe gestern Sabine müller getroffen.`

`Ich habe gestern Sabihne Müller getroffen.`

I experienced this fact to make such taggers close to useless for *realistic* corpora like Tweets, Emails, etc.

This project presents a proof of concept for a different more robust approach. (This might very well be outdated now, a few years later)
I evaluated it on data that is not publicly available and can thus not be provided here, however this approach did outperform
Spacy significantly. Feel free to evaluate your own datasets.

## General Idea
This tagger presents a purely context based approach to NER tagging using a large (untagged) corpus and a small list of 
seed word belonging to the entity to be learned.
Now all instances of the seed list in the corpus get masked, their context will be learned by a MLP
which is then used to identify new entities and append these to the seedlist. This process is repeated multiple times
until the MLP learned most of the contexts where such entities may appear.

### Example
Given some simplified corpus of sentences

* `Hello, my name is Augustus.`
* `Hello, my name is Cleopatra.`
* `Hello, my name is Julia.`
* `I met Julia in town yesterday.`

we want to create a model that is able to tag all Names/Persons by using only a small 
seed list `[Augustus, Cleopatra]`

#### Iteration 1:
##### Masking seed list in all sentences
* `Hello, my name is {MASKED}.`
* `Hello, my name is {MASKED}.`
* `Hello, my name is Julia.`
* `I met Julia in town yesterday.`

##### Training and Predicting
A MLP is trained to predict for every word, how likely it is to be part of the **MASKED** group
only judging by its surrounding context (not the word itself). All words with an assigned probability higher than a certain threshold probability
get added to the seed list. In this case `Julia` in sentence 3 appears in the same context as both masked words and gets a high ranking.
Thus the seed list is extended to `[Augustus, Cleopatra, Julia]`

#### Iteration 2 (Final iteration in this case)
##### Masking seed list in all sentences
* `Hello, my name is {MASKED}.`
* `Hello, my name is {MASKED}.`
* `Hello, my name is {MASKED}.`
* `I met {MASKED} in town yesterday.`

##### Training the MLP
Same as before, an MLP gets trained to predict if a word is MASKED given only its context.

#### Using the Model
If the model is given sentences containing similar context, it is able to pick out words that might belong 
to the desired Entity. 
So if given the sentence `I met Caesar in town yesterday.` the model will most likely return `Caesar`as a NAME/PERSON 
although it has never seen it before.

## Remarks
* The algorithm appears simple but has shown to perform better than the spacy tagger at least for my test dataset.
* One advantage is that the dataset the tagger is used on is also the training set and does **not need to be labeled**.
Only a few seed words are required (The more the better but depending on the use case 10-20 should be enough)
The tagger is thus adjusted to a specific usecase and also learns about common mistakes in that specific set.
* The word itself is not included in judging its entity type, its spelling is thus not important.
* As said before: This is a POC, there are definitely numerous ways of improving it. Maybe its results are just one of many
features to a final NER classifier.
* This approach makes sense for either unknown and erronous corpora or unknown types of entities. FOr largely correct 
corpora and known entities, traditional approaches might work better.


