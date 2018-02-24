#!/bin/sh
awk -F',' '{ print NF }' PollutionOriginal.csv | uniq

# pollution_test/train_transformed have wind converted to integer from categorical
# pollution_test and Pollution_train have final column of next step for pm
# test has 2014 train has everything else

