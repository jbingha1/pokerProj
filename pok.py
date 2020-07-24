#!/usr/local/bin/python3
import re
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd

numRanks = 13
numCards = 52
# All possible hands
hands = [["AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s"],
        ["AKo", "KK", "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s"],
        ["AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s"],
        ["AJo", "KJo", "QJo", "JJ", "JTs", "J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s"],
        ["ATo", "KTo", "QTo", "JTo", "TT", "T9s", "T8s", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s"],
        ["A9o", "K9o", "Q9o", "J9o", "T9o", "99", "98s", "97s", "96s", "95s", "94s", "93s", "92s"],
        ["A8o", "K8o", "Q8o", "J8o", "T8o", "98o", "88", "87s", "86s", "85s", "84s", "83s", "82s"],
        ["A7o", "K7o", "Q7o", "J7o", "T7o", "97o", "87o", "77", "76s", "75s", "74s", "73s", "72s"],
        ["A6o", "K6o", "Q6o", "J6o", "T6o", "96o", "86o", "76o", "66", "65s", "64s", "63s", "62s"],
        ["A5o", "K5o", "Q5o", "J5o", "T5o", "95o", "85o", "75o", "65o", "55", "54s", "53s", "52s"],
        ["A4o", "K4o", "Q4o", "J4o", "T4o", "94o", "84o", "74o", "64o", "54o", "44", "43s", "42s"],
        ["A3o", "K3o", "Q3o", "J3o", "T3o", "93o", "83o", "73o", "63o", "53o", "43o", "33", "32s"],
        ["A2o", "K2o", "Q2o", "J2o", "T2o", "92o", "82o", "72o", "62o", "52o", "42o", "32o", "22"]]

# All simplified positions based on number of players
positions = [['B', 'B', 'E', 'E', 'E', 'M', 'M', 'L', 'L', 'L'],
['B', 'B', 'E', 'E', 'M', 'M', 'L', 'L', 'L'],
['B', 'B', 'E', 'E', 'M', 'M', 'L', 'L'],
['B', 'B', 'E', 'M', 'M', 'L', 'L'],
['B', 'B', 'E', 'M', 'L', 'L'],
['B', 'B', 'E', 'M', 'L'],
['B', 'B', 'M', 'L']]

# Plots a player's range
def plot(data, title):

    matrix = np.array(data)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="autumn_r")
    norm = matplotlib.colors.BoundaryNorm(np.linspace(0, 2, 4), 3)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: ["fold", "call", "raise"][norm(x)])
    cbar_kw = dict(ticks=np.arange(0,2,.99), format=fmt)
    cbar = fig.colorbar(im, **cbar_kw)
    threshold = matrix.max()/2
    textcolors=["black", "white"]

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for i in range(numRanks):
        for j in range(numRanks):
            text = ax.text(j, i, hands[i][j], ha="center", va="center",
            color=textcolors[float(matrix[i][j]) > float(threshold)])

    ax.set_title(title)
    plt.show()

# Create's a players range
def createRange(list):

    indices = []
    f = open("handOrder.txt", "r")
    for line in f.readlines():
        o = line.rstrip()
        for i, row in enumerate(hands):
            for j, column in enumerate(row):
                if column == o:
                    indices.append((i, j))

    a = [[0]*13 for c in range(13)]
    tot = len(list)
    o = 0
    for ra in list:
        r = round(ra*169)
        for i in indices[o:r]:
            a[i[0]][i[1]]=tot
        tot-=1
        o = r
    return a

# This function returns an object of nested dictionaries.
# The order of the structure is name->position->bet->action->frequency
def createObj():
    f = open("allHands.txt", "r")
    hands = f.read().split("chips\n")

    obj = {}

    for a in hands:
        # print("*********")
        b = re.search("[0-9].*won", a, flags=re.DOTALL)
        result = b.group(0) if b else ""
        lines = result.split("\n")
        d = []
        counter = -1
        pos = {}
        last = {}
        bet = 1
        for c in lines:
            if c.endswith("hand"):
                players = int(c[0])
                if players == 1:
                    players = 10
                # if players > 7:
                #     break
                p = positions[10-players]
            elif c.startswith("board"):
                break
            elif re.search("won", c):
                break
            elif len(c.split())>1:
                z = c.split()
                name = z[0]
                action = z[1]
                if counter < len(p)-1:
                    counter += 1
                    pos[name] = p[counter]

                # Cleans up data a little bit
                if action == "paid" or action =="bet" or action=="posted":
                    continue

                # This block ensures that only the final action by the player is recorded
                if name in last:
                    obj[name][pos[name]][last[name][1]][last[name][0]]-=1
                last[name] = (action, bet)

                # Checks to see if the data structure exists. If it does, then it adds 1
                # If it does not it will create the new data structure
                if name in obj:
                    if pos[name] in obj[name]:
                        if bet in obj[name][pos[name]]:
                            if action in obj[name][pos[name]][bet]:
                                obj[name][pos[name]][bet][action] += 1
                            else:
                                obj[name][pos[name]][bet][action] = 1
                        else:
                            obj[name][pos[name]][bet] = {action:1}
                    else:
                        obj[name][pos[name]] = {bet:{action:1}}
                else:
                    obj[name] = {pos[name]:{bet:{action:1}}}
                if action == "raised":
                    bet += 1

                d.append(c)
    return obj

# Plots knn clustered play styles
def plotPlayStyles(obj):
    simp = {}
    for player in obj:
        actions = {"folded":0, "called":0, "raised":0}
        simp[player] = actions
        for position in obj[player]:
            for bet in obj[player][position]:
                for action in obj[player][position][bet]:
                    if action =="checked" or action=="showed":
                        continue
                    simp[player][action]+=obj[player][position][bet][action]
    pl = []
    X = [[],[]]
    for p in simp:
        c = simp[p]["called"]
        r = simp[p]["raised"]
        f = simp[p]["folded"]
        pl.append(p)
        X[0].append(r/(c+r))
        X[1].append((c+r)/(c+r+f))
    pl.append("GTO")
    X[0].append(.55)
    X[1].append(.25)

    d = {'ag':X[0], 'hand':X[1]}
    df = pd.DataFrame(d)

    distortions = []
    for k in range(1,10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        distortions.append(sum(np.min(cdist(df, kmeans.cluster_centers_, 'euclidean'), axis=1))/df.shape[0])

    plt.plot(range(1,10), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df)
    y_means = kmeans.predict(df)

    fig, ax = plt.subplots()
    ax.scatter(X[0], X[1], c=y_means, cmap='viridis')

    for i, playa in enumerate(pl):
        ax.annotate(playa, (X[0][i], X[1][i]))

    ax.set_title("Play Style")
    ax.set_ylabel("% Hands played")
    ax.set_xlabel("Agressiveness in Hands Played")
    plt.show()

    return simp

# Plots players' odds of playing hands based on their position
def plotPositionPercents(obj):
    X = ["Early Position", "Middle Position", "Late Position"]

    Y = {}
    y = {}
    for name in obj:
        Y[name] = {}
        y[name] = []
        for pos in obj[name]:
            Y[name][pos] = {}
            for bet in obj[name][pos]:
                for action in obj[name][pos][bet]:
                    if action in Y[name][pos]:
                        Y[name][pos][action] += obj[name][pos][bet][action]
                    else:
                        Y[name][pos][action] = obj[name][pos][bet][action]
            try:
                Y[name][pos]['total'] = (Y[name][pos]['raised'] + Y[name][pos]['called']) / (Y[name][pos]['raised'] + Y[name][pos]['called'] + Y[name][pos]['folded'])
            except:
                pass
        try:
            y[name].append(Y[name]['E']['total'])
            y[name].append(Y[name]['M']['total'])
            y[name].append(Y[name]['L']['total'])
            plt.plot(X, y[name], label=name)
        except:
            y.pop(name)

    av = [0,0,0]
    for n in y:
        av[0] += y[n][0]
        av[1] += y[n][1]
        av[2] += y[n][2]

    av[0] = av[0]/len(y)
    av[1] = av[1]/len(y)
    av[2] = av[2]/len(y)

    plt.plot(X, av, label="Average", linestyle="dashed")
    plt.plot(X, [.2, .25, .3], label="GTO", linestyle="dashed")

    plt.title("Hand percentages played by position")
    plt.ylabel("% Hands played")
    plt.legend()
    plt.show()
    print(y)

# Plots players' odds of playing hands based on number of players in a hand
def plotPlayerPercents():
    f = open("allHands.txt", "r")
    hands = f.read().split("chips\n")

    obj = {}

    for a in hands:
        # print("*********")
        b = re.search("[0-9].*won", a, flags=re.DOTALL)
        result = b.group(0) if b else ""
        lines = result.split("\n")
        last = {}
        for c in lines:
            if c.endswith("hand"):
                players = int(c[0])
                if players == 1:
                    players = 10
            elif c.startswith("board"):
                break
            elif re.search("won", c):
                break
            elif len(c.split())>1:
                z = c.split()
                name = z[0]
                action = z[1]

                # Cleans up data a little bit
                if action == "paid" or action =="bet" or action=="posted":
                    continue

                # This block ensures that only the final action by the player is recorded
                if name in last:
                    obj[name][players][last[name]]-=1
                last[name] = action

                # Checks to see if the data structure exists. If it does, then it adds 1
                # If it does not it will create the new data structure
                if name in obj:
                    if players in obj[name]:
                        if action in obj[name][players]:
                            obj[name][players][action] += 1
                        else:
                            obj[name][players][action] = 1
                    else:
                        obj[name][players] = {action:1}
                else:
                    obj[name] = {players:{action:1}}

    for n in obj:
        obj[n]["0"] = [[],[]]
        k = sorted(int(a) for a in obj[n].keys())
        for players in k:
            try:
                num = (obj[n][players]["called"]+obj[n][players]["raised"])
                denom = (obj[n][players]["folded"]+obj[n][players]["raised"]+obj[n][players]["called"])
                if denom < 35:
                    continue
                obj[n]["0"][0].append(num/denom)
                obj[n]["0"][1].append(players)
            except:
                pass
        if len(obj[n]["0"][0])>3:
            plt.plot(obj[n]["0"][1], obj[n]["0"][0], label=n)

    plt.xlabel("Players in the hand")
    plt.ylabel("% Hands played")
    plt.title("% of hands played based on # of players in hand")
    plt.legend()
    plt.show()




# Uncomment below lines to plot mtgallo2's range
# obj = createObj()
# o = plotPlayStyles(obj)
# f = o["mtgallo2"]["folded"]
# r = o["mtgallo2"]["raised"]
# c = o["mtgallo2"]["called"]
# a = createRange([r/(f+r+c), (r+c)/(f+r+c)])
# plot(a, "mtgallo2 range")

# Uncomment below lines to plot jonnybows's range
# obj = createObj()
# o = plotPlayStyles(obj)
# f = o["mtgallo2"]["folded"]
# r = o["mtgallo2"]["raised"]
# c = o["mtgallo2"]["called"]
# a = createRange([r/(f+r+c), (r+c)/(f+r+c)])
# plot(a, "mtgallo2 range")

# Uncomment the below lines to plot players' hand % played vs number of players in hand
# plotPlayerPercents()

# Uncomment the below lines to plot players' hand % played vs player position
# obj = createObj()
# plotPositionPercents(obj)

# Uncomment the below lines to plot players' players' play styles
# obj = createObj()
# plotPlayStyles(obj)
