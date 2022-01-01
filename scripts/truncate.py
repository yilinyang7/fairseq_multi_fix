import fileinput

for s in fileinput.input():
    s = s.rstrip().split(' ')[:254]
    print(' '.join(s))

