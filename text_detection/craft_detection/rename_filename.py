import os
from tqdm import tqdm


for d in os.listdir('dataset'):
    print('d')
    i = 1
    for f in tqdm(os.listdir(os.path.join('dataset',d))):
        os.rename(os.path.join('dataset',d,f),os.path.join('dataset',d,d+'_'+str(i)+'.jpg'))
        i += 1
    
    
        