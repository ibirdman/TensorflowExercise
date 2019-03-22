dict={"name":"zs","age":18,"city":"深圳","tel":"1362626627"}
res = sorted(dict.items(), key=lambda k:k[0]);
new_dict = {};
for x in res:
    new_dict[x[0]]=x[1];

print(new_dict);