a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def fn(x):
    return x % 2 == 1;

res = list(map(fn, a));
print(res);


b = [i for i in a if i % 2 == 1];
print(b);