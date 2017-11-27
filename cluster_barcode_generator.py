import csv
import re
import sys

def raw_values_from_csv(filename):
    values = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            values.append(row)
    return values

index_list_files = sys.argv[1:]

index_lists = []
largest_index = 0
for file in index_list_files:
    str_values = raw_values_from_csv(file)
    int_values = [int(s[0]) for s in str_values]
    m = max(int_values)
    if(m>largest_index):
        largest_index = m
    index_lists.append(int_values)

barcodes = []
for x in range(0,largest_index+1):
    barcode = ""
    for n,il in enumerate(index_lists):
        found = False
        for i in il:
            if(x == i):
                found = True
        if(found):
            barcode = barcode + str(n+1)
        else:
            barcode = barcode + "_"
    barcodes.append(barcode)

for b in barcodes:
    print(b)

