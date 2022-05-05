l1=[]
l2=[]
print("first sentence")
p1=input("Enter predicate ")
while True:
    st = input("enter arguement ")
    if not st:
        break
    l1.append(st)
p2=input("second predicate ")
while True:
    st = input("enter arguement ")
    if not st:
        break
    l2.append(st)
#rule 1
if(p1!=p2):
    print("Predicate not same")
elif(len(l1)!=len(l2)):
    print("number of arguements not same")
else:
    for i in range(len(l1)):
        print('({}|{})'.format(l1[i],l2[i]))




