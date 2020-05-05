from __main__ import *

def precisionget(predicted,labels,indi, tots):
    p=int(predicted)
    l=int(labels)

    index=int(labels)
    indextots=index
    t= tots[index]
    i= indi[index]
    if int(predicted) == int(labels):
        indi[index]+=1

    else:
        None
        # print("not equal")

    tots[indextots]+=1

    PR=np.zeros(len(indi))

    for i in range(len(indi)):
        PR[i]=100*indi[i]/tots[i]

    NanSpot=np.argwhere(np.isnan(PR))
    PR=np.delete(PR,NanSpot)

    return indi,tots,PR
