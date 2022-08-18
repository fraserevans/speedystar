from astropy.table import Table, vstack



def catstack(catlist,path_to_write):
    '''
    -- Concatenates many fits tables, writes result to file in path_to_write
    '''

    t1 = Table.read(catlist[0],format='fits')

    for cat in catlist[1:]:
        t2 = Table.read(cat,format='fits')
        t1 = vstack([t1,t2])

    t1.write(path_to_write,overwrite=True)

        

