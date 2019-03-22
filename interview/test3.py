s = 'abaccddefgh';
result = set(s);
s = list(result)

'''
for x in s:
    exist = x in result;
    if not exist:
        result += x;
'''
s.sort();
res = "".join(s);
print(res);


