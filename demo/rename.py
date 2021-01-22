import os
import glob

path = './demo/OFtest/*.png'

i = 0

flist = sorted(glob.glob(path))

for file in flist:
    os.rename(file, './demo/OFtest/' + str(i).zfill(6) + '.png')
    i+=1
