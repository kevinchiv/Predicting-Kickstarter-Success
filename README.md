# Predicting Kickstarter Success

## Description: 
Kickstarter is a global crowdfunding platform that helps independent creators finance their projects. What makes Kickstarter noteworthy and powerful is its community. Specifically, the community's ability to turn campaigns into million dollar success stories. 

## Objective: 
Each year, several projects fail.

The goal of this project is to help independent creators pinpoint factors that could help or hinder a potential campaign.

## Methodology: 
The dataset I used was obtained from https://www.kaggle.com/kemical/kickstarter-projects/data#
Here are the features I worked with:
- title_length, the number of words in the campaign title		
- min_45_backers, whether the campaign has a minimum of 45 backers or not
- goal, the goal amount the Kickstarter project team is aiming for
- project_duration, the campaign duration
- design, whether the main category of the Kickstarter campaign is design
- film_video, whether the main category of the Kickstarter campaign is film & video
- games, whether the main category of the Kickstarter campaign is games	
- technology, whether the main category of the Kickstarter campaign is technology

I used classification to design a machine learning app.

The machine learning model was trained on Kickstarter data from 2009 to 2017. 
The model uses F2 i.e. it minimizes the number of false positives and false negatives, with false negatives being weighted more.

A false positive is campaign that was predicted to succeed, but would have failed. 
Meanwhile, a false negative is a campaign that was predicted to failed, but would have succeeded.

False negatives are weighted more because potentially successful projects should recieve the funding they need and deserve. Campaign success is a mutual goal for creators, backers, and Kickstarter itself.


## Results: <br>
The best model was XGB Classifier, but I chose to use SGD Classifier for the speed, scalability, and model interpretability.
In terms of F2, SGD Classifer had a score of 0.822 on train and 0.823 on test.

If we ignore the main categories, the results indicate that on average: <br>
- having a minimum of 45 backers is the most important 
- having a shorter project duration is crucial, specifically 30 days or less
- have a reasonably long i.e. informative and interesting title is significant
- having a reasonable goal amount helps

The app is currently hosted at: 
http://kickstarter-success-predictor.herokuapp.com/
