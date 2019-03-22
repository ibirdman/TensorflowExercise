num = [1, 4, -5, 10, -7, 2, 3, -1]

a = [i**2 for i in num if i > 0]
print(a);

num = [1, 4, -5, 10, -7, 2, 3, -1]
filtered_and_squared = (x**2 for x in num if x > 0)
print(list(filtered_and_squared))
