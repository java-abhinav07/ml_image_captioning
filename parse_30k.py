import csv
import os

with open("./data/flickr30k_images/caps.txt", 'w') as file:
    with open("./data/flickr30k_images/results.csv") as f:
        names = csv.reader(f)
        for name in names:
            # print(name[0])
            sp = name[0].split("|")
            im = sp[0]
            cap = sp[-1]
            print(im, cap)
            file.write(str(im))
            file.write(",")
            file.write(str(cap))
            file.write("\n")
