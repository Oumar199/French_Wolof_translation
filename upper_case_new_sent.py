import pandas as pd
sentences = pd.read_csv("wolof-translate/wolof_translate/data/sentences/wolof_french.csv")
french = sentences['french']
wolof = sentences['wolof']
new_french = []
new_wolof = []

for sent in french:
     letters = [sent[0], sent[1]]
     for l in sent[2:]:
             if letters[-1] in ['?', '!', '.'] or letters[-2] in ['?', '.', '!'] and l.isupper():
                     letters.append(l.upper())
             else:
                     letters.append(l)
     new_french.append("".join(letters))
 
for sent in wolof:                                                            
     letters = [sent[0], sent[1]]
     for l in sent[2:]:
             if letters[-1] in ['?', '!', '.'] or letters[-2] in ['?', '.', '!'] and l.isupper():
                     letters.append(l.upper())                                 
             else:
                     letters.append(l)
     new_wolof.append("".join(letters))

new_sents = pd.DataFrame({'french': new_french, 'wolof': new_wolof})
new_sents.to_csv('wolof-translate/wolof_translate/data/sentences/wolof_french.csv', index = False)
# print(new_sents.head(10))

