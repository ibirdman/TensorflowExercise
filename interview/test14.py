import os

def tree(top):
    for path, names, fnames in os.walk(top):
        print(path);
        for fname in names:
            yield os.path.join(path, fname)

'''
for name in tree('/home/CORPUSERS/28848747/gwas/icons'):
    print(name);'''

a = list(tree('/home/CORPUSERS/28848747/gwas/icons'));
#print(a);