# Stock-Price-Predictor-using-LSTM

This repo contains 2 files both of which serve a similar purpose, to predict stock prices. The first file is a direct copy of code I found online on a blog, which predicts the price of the next day. The other file is my attempt to try and make the model predict the file for multiple days.

# File 1: Stock Price Predictor using LSTM - original

The code was originally provided by Abhishek Shaw, on a blog on Medium. (link: https://medium.com/@abhishekshaw020/python-project-building-a-real-time-stock-market-price-prediction-system-6ce626907342)


A copy of a short term Stock Price predictor that uses LSTM. This code was written with the intention of playing around with a few different libraries, and also serves as being my first developer-ended experience with machine learning in general. The code was difficult to understand, and I still haven't completely gone through and understood all of it. It marks the beginning of my journey into studying ML and implementing it myself on mini-projects to help me on my journey to being a ML maestro from an ML amateur.

I played around with a few values, and I saw the importance of selecting the right number of epochs, batch sizes etc.

# File 2: Stock Price Predictor using LSTM - updated

The code is largely similar to the first, except I tried to see if I could get it to predict prices farther into the future. Spoiler alert: I can't.

The code runs, and it gives initially believable values. As the number of days into the future increases, so does the error. In fact, it only either moves upwards or downwards, depending on how the training data ends. This is in no way an industry-ready model, but it still seems cool to have played around with and made a few changes of my own! 
