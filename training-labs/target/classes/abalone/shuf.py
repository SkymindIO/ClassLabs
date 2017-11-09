# read from standard input and write to standard output for simplicity
import sys
import fileinput
import random
 
lines = [ line for line in fileinput.input() ]
random.shuffle(lines)
for line in lines:
    print line
