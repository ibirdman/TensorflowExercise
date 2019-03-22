def f(x, l=[]):
    for i in range(x):
        l.append(i * i)
    print(l);

f(2);
f(3);