## regex.py
##
## A regex sandbox to test regex functionality
##
## Functions to test:
## match, search, sub, subn, split, findall, finditer, compile, purge, escape, template
##
## Flags:
## I (IGNORECASE), L (LOCALE), M (MULTILINE), S (DOTALL), X (VERBOSE), U (UNICODE)
##
## 

'''
import re

test1 = 'This is the story of a man named Kevin Mandich. He lived at 1849 Bush St., San Francisco, CA. He moved to San Francisco on January 8, 2014, and has been living at his current address since July 31, 2015. His e-mail address is kevinmandich@gmail.com. His girlfriend\'s name is Rumi Yokota.'

test2 = 'aaaabbbbcccdddcccbbbbaaaa abcdedcba abcde aBcDeEdCbA'

print '\n','Test Text 1:\n'
print test1,'\n'

print '\n','Test Text 2:\n'
print test2,'\n\n'

testEscaped = re.escape(test1)
print testEscaped,'\n\n'

cityRegex = re.compile(r'San\s+([A-Z][a-z]*)')
hisRegex = re.compile(r'(his|His)')

allCities = cityRegex.findall(test1)
print allCities,'\n\n'

his = hisRegex.findall(test1)

groupRegex = re.compile(r'a(b(c(d(e)d)c)b)a')

g = groupRegex.search(test2)
'''

def reverse(string):

    if len(string) == 1:
        return string

    return reverse(string[1:]) + string[0]


def read_in_chunks(fileObject, chunkSize=1024):

    while True:
        data = fileObject.read(chunkSize)
        if not data:
            break
        yield data

f = open('sample_big_file.dat', 'rb')

for chunk in read_in_chunks(f, 528):
    print chunk
    # process_data(chunk)


test = 'abcdefgh'

print test
print reverse(test)




































