sum = 0.0
sum2 = 0.0
# main loop
while True:
    try:
        data = float(input("input number:"))
    except ValueError: # end of input
        break
    sum += data
    sum2 += data*data
    print("{:.15f} {:.15f}".format(sum, sum2))