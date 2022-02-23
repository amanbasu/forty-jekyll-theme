---
layout: post
title: Zomato data visualization
description: Analysing restaurant data using various visualizations
image: assets/images/zomatoviz.png
nav-menu: true
---

View notebook [here](https://nbviewer.org/github/amanbasu/zomato-data-visualization/blob/master/viz-notebook.ipynb).

The goal of this project is to create visualizations that thoroughly analyzes restaurant directory information in order to answer specific food-related queries. As part of this framework, we hope to add rich visualizations based on restaurant meta-data such as location, pricing, ratings, and cuisine to aid in the resolution of various issues.

Our solution can help solve the problems for two buckets of people -

## Customer/Foodie

Neha is a college student, studying Liberal Arts at the City University of Delhi. In about a week or so, she will be celebrating her 26th birthday. She wants to throw a big bash for her friends and family. She is overwhelmed by the number of options displayed by restaurant directories. She came to us asking for a study of restaurants in her locality based on the best cuisine, best bargains, and ratings as she didn’t want to compromise on her big day.

Here are some questions Neha asked us:

1. Which is the best neighborhood with the top restaurants in her city? 2. Which locality is the best, if she wants to keep the bill low?
3. What is a cuisine that is really popular in her neighborhood?
4. Which would be the best locality to go to for ‘X’ cuisine?
5. Does paying more for food mean she will have a better experience?

## Restaurant owner

Karan is a restaurant owner in Bangalore, India. He opened his restaurant in 2017, in one of the prime locations in Bangalore. In the past few months, Karan is losing business while there has been an increase in restaurants around him. Karan then came to us asking for a detailed analysis of the neighboring restaurants around him - including but not limited to the rating distribution, price ranges, and also a partition by the restaurant cuisines. He will then use this analysis to further make better business decisions on whether his pricing strategy and cuisine is ideal or not.

Here are some questions Karan asked us:

1. Is the location for my restaurant congenial - in terms of pricing, cuisine, and density? 2. How much do restaurants charge on average in this given locality?
3. What are the restaurants that are posing as the prime competitor for me?
4. Which cuisine is most in-demand in this locality?

We exhibit various visualizations of restaurant data in India throughout the study to discover answers to some of the above queries. 

## Dataset

This dataset has been pulled from Kaggle [Zomato India Restaurants(2 lakh restaurants data)](https://www.kaggle.com/ngokulakannan/zomato-india-restaurants2-lakh-restaurants-data). It contains 224,520 restaurant and 17 columns/attributes. The major attributes are restaurant name, city, rating, latiitude, longitude, famous food, and cuisine. This will be the primary dataset for all the statistical analysis and visualizations.
