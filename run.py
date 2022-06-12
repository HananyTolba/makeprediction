from matplotlib.ft2font import HORIZONTAL
import makeprediction as mp
HORIZONTAL = "''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"
print(HORIZONTAL.center(80))

with open("assets/logo-ascii-art.txt",'r') as f:
    for line in f:
        line = line.center(90)
        print(repr(line.replace("#","@").replace('\n',' ')))
welcome = f'Welcome to {mp.__name__}: {mp.version.__version__} \U0001F600'  
print(welcome.center(80))
print(HORIZONTAL.center(80))

