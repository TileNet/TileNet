import random

base1 = 9
base2 = 10
f = open("data/%d-%d.txt" % (base1, base2), "w+")

def numberToBase(n, b):
    if n == 0:
        return ['0']
    digits = []
    while n:
        digits.append(str(int(n % b)))
        n //= b
    return digits[::-1]

def writeLine(i):
    input = numberToBase(i, base1)
    input = "".join(input)
    output = numberToBase(i, base2)
    output = "".join(output)
    line = input + "," + output + "\n"
    f.write(line)

for i in range(1,base2+1):
    writeLine(i)

for len in range(5):
    for count in range(1000):
        i = random.randint(base1**len-1, base1**(len+1)-1)
        writeLine(i)

for len in range(2000):
    i = random.randint(base1**len-1, base1**(len+1)-1)
    writeLine(i)

f.close()
