import re

a = "not 404 found 张三 99 深圳"
b = a.split(' ');
print(b);
res = re.findall('\d+|[A-Za-z]+', a);
print(res);
for c in res:
    b.remove(c);

print(b);