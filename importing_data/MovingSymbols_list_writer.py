dir_file_train = '/netscratch/arenas/dataset/moving_symbols/MovingSymbols2_trainlist.txt'
dir_file_test = '/netscratch/arenas/dataset/moving_symbols/MovingSymbols2_testlist.txt'
movements = ['Horizontal', 'Vertical']

trainfile = open(dir_file_train, 'w')
for i, mov in enumerate(movements):
    for j in range(5000):
        content = trainfile.write('%s/%s_video_%s.avi\n' % (mov, mov, j+1))
trainfile.close()

testfile = open(dir_file_test, 'w')
for k, mov in enumerate(movements):
    for j in range(500):
        content = testfile.write('%s/%s_video_%s.avi\n' % (mov, mov, j+1))
testfile.close()
