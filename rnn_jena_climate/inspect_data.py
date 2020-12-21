import os


data_dir = '/Users/iliyanslavov/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
file = open(fname)
data = file.read()
file.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(f'Header: {header}')
print(f'Lines: {len(lines)}')
