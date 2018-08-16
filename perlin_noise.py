import numpy as np

###############################3
#### Generating smooth noise
def perlin(x,y=None,seed=0):
    '''
    Perlin noise is a smooth noise.
    '''
    #
    if y is not None :
        if x.ndim ==1 & y.ndim ==1 :
            xx,yy = np.meshgrid(x,y)
        else :
            if x.ndim == y.ndim:
                xx = x.copy()
                yy = y.copy()
            else : #graceful exit
                print('x and y do not have the same dimensions')
                return -1
    else:
        y_ = np.array([0,1])
        xx,yy = np.meshgrid(x,y_)
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = xx.astype(int)
    yi = yy.astype(int)
    # internal coordinates
    xf = xx - xi
    yf = yy - yi
    # fade factors
    u = fade_perlin(xf)
    v = fade_perlin(yf)
    # noise components
    n00 = gradient_perlin(p[p[xi]+yi],xf,yf)
    n01 = gradient_perlin(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient_perlin(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient_perlin(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp_perlin(n00,n10,u)
    x2 = lerp_perlin(n01,n11,u) # FIX1: I was using n10 instead of n01
    if y is None : # 1d
        return lerp_perlin(x1,x2,v)[0] # FIX2: I also had to reverse x1 and x2 here
    else :
        return lerp_perlin(x1,x2,v)

def lerp_perlin(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade_perlin(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient_perlin(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y



