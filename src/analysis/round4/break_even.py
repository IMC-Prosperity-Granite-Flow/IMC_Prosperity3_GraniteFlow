#%%
def break_even(config: list, target: float):
    return config[0]/target - config[1]

data = [[80, 6], 
        [50, 4], 
        [83, 7], 
        [31, 2], 
        [60, 4], 
        [89, 8], 
        [10, 1], 
        [37, 3], 
        [70, 4], 
        [90, 10],
        [17, 1],
        [40, 3],
        [73, 4],
        [100, 15],
        [20, 2],
        [41, 3],
        [79, 5],
        [23, 2],
        [47, 3],
        [30, 2]]

flag = True
for i in range(1000, 200, -1):
    summation = 0
    if flag:
        for item in data:
            summation += break_even(item, i/100)
            if summation > 100:
                print(summation)
                flag = False
    else:
        break

print(i/100)
# %%
dist = [break_even(item, 5.65) for item in data]

#%%
delta = []
for i in range(len(dist)):
    # delta (first order derivative) = -amount/(inhabitants + dist)^2
    amount = data[i][0]
    inhabitants = data[i][1]
    delta.append(-amount/(inhabitants + dist[i])**2)