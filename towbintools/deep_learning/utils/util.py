def divide_batch(l, n): 
        for i in range(0, l.shape[0], n):  
            yield l[i:i + n,::] 