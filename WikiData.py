import sys
import numpy as np
import pandas as pd
import re
import pickle as pickle

class WikiData:
    def load(self):
        filename = "wikiData.txt"
        self.df = pd.read_csv(filename,sep='\t',encoding ='latin-1',header=0)

    def save_object(self, obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        self.nCols = len(self.df.columns)
    def generateEntities(self):
        self.entities = {}
        for index, row in self.df.iterrows():
            row['human'] = row['human'].replace('http://www.wikidata.org/entity/','')
            row['child'] = row['child'].replace('http://www.wikidata.org/entity/','')
            row['spouse'] = row['spouse'].replace('http://www.wikidata.org/entity/','')
            row['sibling'] = row['sibling'].replace('http://www.wikidata.org/entity/','')
            row['father'] = row['father'].replace('http://www.wikidata.org/entity/','')
            row['mother'] = row['mother'].replace('http://www.wikidata.org/entity/','')

            self.entities[row['human']] = row['humanLabel']
            self.entities[row['child']] = row['childLabel']
            self.entities[row['spouse']] = row['spouseLabel']
            self.entities[row['sibling']] = row['siblingLabel']
            self.entities[row['father']] = row['fatherLabel']
            self.entities[row['mother']] = row['motherLabel']

        fout = "entities.txt"
        fo = open(fout, "w", encoding='latin-1')
        for k, v in self.entities.items():
            fo.write(v+ '\n')

        self.save_object(self.entities,'entities_map.pkl')

    def generatePredicates(self):
        self.predicates = {}
        self.predicates['P22'] = 'fatherIs'
        self.predicates['P25'] = 'motherIs'
        self.predicates['P3373'] = 'siblingIs'
        self.predicates['P26'] = 'spouseIs'
        self.predicates['P40'] = 'childIs'

        fout = "predicates.txt"
        fo = open(fout, "w", encoding='latin-1')
        for k, v in self.predicates.items():
            fo.write(v+ '\n')

    def generatePositiveTriplets(self):
        self.posTriplets = set()
        self.posSentences = ""
        fout = "positiveTriplets.txt"
        fo = open(fout, "w", encoding='latin-1')
        for index, row in self.df.iterrows():
            childTriplet = row['human']+' P40 ' + row['child'] + ' 1'
            childSentence = row['humanLabel']+' P40 ' + row['childLabel']
            fo.write(childTriplet+ '\n')
            self.posTriplets.add(childTriplet)
            self.posSentences += childSentence + '\n'

            spouseTriplet = row['human']+' P26 ' + row['spouse'] + ' 1'
            spouseSentence = row['humanLabel']+' P40 ' + row['spouseLabel']
            fo.write(spouseTriplet+ '\n')
            self.posTriplets.add(spouseTriplet)

            siblingTriplet = row['human']+' P3373 ' + row['sibling'] + ' 1'
            fo.write(siblingTriplet+ '\n')
            self.posTriplets.add(siblingTriplet)

            fatherTriplet = row['human']+' P22 ' + row['father'] + ' 1'
            fo.write(fatherTriplet+ '\n')
            self.posTriplets.add(fatherTriplet)

            motherTriplet = row['human']+' P25 ' + row['mother'] + ' 1'
            fo.write(motherTriplet+ '\n')
            self.posTriplets.add(motherTriplet)

    def generateNegativeTriplets(self):
        self.negTriplets = set()
        fout = "negativeTriplets.txt"
        fo = open(fout, "w", encoding='latin-1')
        for t in self.posTriplets:
            strings = t.split()
            subject = strings[0]
            predicate = strings[1]
            tobject = strings[2]
            max_for_sp_pair=2
            count= 0
            for ek, ev in self.entities.items():
                triplet = subject+' '+predicate+' '+ek
                if triplet+' 1' not in self.posTriplets and ek!=subject:
                    self.negTriplets.add(triplet+' -1')
                    if count < max_for_sp_pair:
                        count=count+1
                    else:
                        break
        for t in self.negTriplets:
            fo.write(t+ '\n')

    def addStringsToVocab(self, strings):
        for s in strings:
            word = ''.join(w for w in re.split(r"\W", s) if w)
            self.vocab+= word+ " "

    def generateVocabulary(self):
        self.unique_words = set()
        self.vocab = ""
        self.entities = {}
        for index, row in self.df.iterrows():
            human_strings = row['humanLabel'].split()
            self.addStringsToVocab(human_strings)
            self.unique_words |= set(human_strings)
            child_strings = row['childLabel'].split()
            self.addStringsToVocab(child_strings)
            self.unique_words |= set(child_strings)
            spouse_strings = row['spouseLabel'].split()
            self.addStringsToVocab(spouse_strings)
            self.unique_words |= set(spouse_strings)
            sibling_strings = row['siblingLabel'].split()
            self.addStringsToVocab(sibling_strings)
            self.unique_words |= set(sibling_strings)
            father_strings = row['fatherLabel'].split()
            self.addStringsToVocab(father_strings)
            self.unique_words |= set(father_strings)
            mother_strings = row['motherLabel'].split()
            self.addStringsToVocab(mother_strings)
            self.unique_words |= set(mother_strings)

        fout = "vocab.txt"
        fo = open(fout, "w", encoding='latin-1')
        fo.write(self.vocab+ '\n')
        fout = "unique.txt"
        fo = open(fout, "w", encoding='utf8')
        for s in self.unique_words:
            fo.write(''.join(w for w in re.split(r"\W", s) if w)+'\n')





wikiData = WikiData()
wikiData.load()
wikiData.generateEntities()
wikiData.generatePredicates()
wikiData.generatePositiveTriplets()
wikiData.generateNegativeTriplets()
wikiData.generateVocabulary()


