import unicodecsv as csv
with open('tSNE.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow([1, 2,'oiy'])