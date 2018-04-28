'''
Created on 12 apr. 2018

@author: Sara Boanca
'''
import matplotlib.pyplot as plt
import numpy as np
from random import randint, random, shuffle, choice
from copy import deepcopy

class Problem:
    def __init__(self, fileName, fileNameWords, fileNameParameters):
        self.__fileName = fileName
        items = []
        items = self.readFromFile(self.__fileName)
        self.__lines = items[0]
        self.__columns = items[1]
        self.__blanks = items[2]
        self.__matrix = items[3]
        self.__slots = items[4]
        words = []
        words = self.readFromFileWords(fileNameWords)
        self.__words = words
        parameters = self.readFromFileParameters(fileNameParameters)
        self.__noEpoch = parameters[0]
        self.__noAnts = parameters[1]
        self.__alpha = parameters[2]
        self.__beta = parameters[3]
        self.__rho = parameters[4]
        self.__q0 = parameters[5]
        
    def getNoEpoch(self):
        return self.__noEpoch
    
    def getNoAnts(self):
        return self.__noAnts
    
    def getAlpha(self):
        return self.__alpha
    
    def getBeta(self):
        return self.__beta
    
    def getRho(self):
        return self.__rho
    
    def getQ0(self):
        return self.__q0
    
    def getLines(self):
        return self.__lines
    
    def getColumns(self):
        return self.__columns
    
    def getBlanks(self):
        return self.__blanks
    
    def getMatrix(self):
        return self.__matrix
    
    def getSlots(self):
        return self.__slots
    
    def getWords(self):
        return self.__words
    
    def getIndividualSize(self):
        return len(self.__slots)
    
    def stringMatrix(self):
        matrix = self.__matrix
        s = ''
        for i in range(self.__lines):
            for j in range(self.__columns):
                m = matrix[i][j]
                s += str(m)
                s += ' '
            s += "\n"
        return s
    
    def readFromFileParameters(self, fileNameParameters):
        parameters = []
        try:
            fd = open(fileNameParameters, 'r')
            noEpoch = int(fd.readline())
            parameters.append(noEpoch)
            noAnts = int(fd.readline())
            parameters.append(noAnts)
            alpha = float(fd.readline())
            parameters.append(alpha)
            beta = float(fd.readline())
            parameters.append(beta)
            rho = float(fd.readline())
            parameters.append(rho)
            q0 = float(fd.readline())
            parameters.append(q0)
        except IOError:
            print("error")
            
        return parameters
        
    
    def readFromFileWords(self, fileName):
        words = []
        try:
            fd = open(fileName, 'r')
            words = fd.readline().split(',')
        except IOError:
            print("error")
        return words
    
    def readFromFile(self, fileName):
        try:
            fd = open(fileName, 'r')
            lin = int(fd.readline())
            col = int(fd.readline())
            blanks = int(fd.readline())
            myMatrix = [[0 for x in range(lin)] for y in range(col)]
            for line in fd:
                (k, v) = line.split(' ')
                i1 = int(k)
                i2 = int(v)
                myMatrix[i1][i2] = '#'
            
        except IOError:
            print("error")
    
        list=[]
        coloana = 0
        i = 0
        nrCrt = 0
        while i < lin:
            while myMatrix[i][coloana] == '#':
                coloana += 1
                if coloana == col:
                    i += 1
                    coloana = 0
            retCol = coloana
            length = 0
            while coloana < col and myMatrix[i][coloana] != '#':
                length += 1
                coloana += 1
            if length > 1:
                myTuple = [nrCrt, 0, i, retCol, length]
                nrCrt += 1
                list.append(myTuple)
            if coloana == col:
                i += 1
                coloana = 0
        
        linie = 0
        j = 0
        while j < col:
            while myMatrix[linie][j] == '#':
                linie += 1
                if linie == lin:
                    j += 1
                    linie = 0
            retLin = linie
            length = 0
            while linie < lin and myMatrix[linie][j] != '#':
                length += 1
                linie += 1
            if length > 1:
                myTuple = [nrCrt, 1, retLin, j, length]
                nrCrt += 1
                list.append(myTuple)
            if linie == lin:
                j += 1
                linie = 0
        
        return [lin, col, blanks, myMatrix, list]

class Ant:
    def __init__(self, problem):
        self.problem = problem
        self.size = problem.getIndividualSize()
        self.path = [randint(0,self.size-1)]
        self.matrix = problem.getMatrix()
        self.slots = problem.getSlots()
        self.words = problem.getWords()
        self.lines = problem.getLines()
        self.columns = problem.getColumns()
        
    def getSize(self):
        return self.size
    
    def getPath(self):
        return self.path
    
    def getNextIndex(self):
        return len(self.path)
    
    def fillCrossword(self):
        position = self.getNextIndex()
        matrix = deepcopy(self.matrix)
        words = self.words
        for i in range(position):
            slotNumber = self.path[i]
            slot = self.slots[slotNumber]
            if slot[1] == 0:
                x = slot[2]
                y = slot[3] - 1
                length = 0
                allowed = True
                while y < self.columns:
                    y += 1
                    while length < len(words[i]) and allowed == True:
                        if matrix[x][y] == 0:
                            matrix[x][y] = words[i][length]
                            y += 1
                            length += 1
                        elif matrix[x][y] == "#":
                            allowed = False
                            break
                        else:
                            if words[i][length] != matrix[x][y]:
                                matrix[x][y] = words[i][length]
                                y += 1
                                length += 1
                            else:
                                matrix[x][y] = words[i][length]
                                y += 1
                                length += 1 
                        if y == self.columns and length < len(words[i]):
                            allowed = False
                            break
            else:
                y = slot[3]
                x = slot[2] - 1
                length = 0
                allowed = True
                while x < self.lines:
                    x += 1
                    while length < len(words[i]) and allowed == True:
                        if matrix[x][y] == 0:
                            matrix[x][y] = words[i][length]
                            x += 1
                            length += 1                                              
                        elif matrix[x][y] == "#":
                            allowed = False
                            break
                        else:
                            if words[i][length] != matrix[x][y]:
                                matrix[x][y] = words[i][length]
                                x += 1
                                length += 1                                                     
                            else:
                                matrix[x][y] = words[i][length]
                                x += 1
                                length += 1                                                    
                        if x == self.lines and length < len(words[i]):
                            allowed = False
                            break
        return matrix
    
    def remainingSlots(self):
        remaining = []
        remainingSlots = []
        for i in range(self.size):
            if i not in self.path:
                remaining.append(i)
        for slot in self.slots:
            if slot[0] in remaining:
                remainingSlots.append(slot)
        return remainingSlots
    
    def filterSlotsByLength(self, length):
        slots = self.remainingSlots()
        filtered = []
        for slot in slots:
            if slot[4] == length:
                filtered.append(slot)
        return filtered
    
    def nextWordIndex(self, a):
        for i in range(len(self.path)):
            if self.path[i] == a:
                return i + 1  
    
    def nextMoves(self):
        words = self.words
        nextPossibleMoves = []
        remainingSlots = []
        remainingSlots = self.remainingSlots()
        index = self.getNextIndex()
        if index >= self.size:
            return []
        length = len(self.words[index])
        goodLengthSlots = self.filterSlotsByLength(length)
        if goodLengthSlots == []:
            return goodLengthSlots
        else:
            for slot in goodLengthSlots:
                matrix = self.fillCrossword()
                accepted = True
                if slot[1] == 0:
                    x = slot[2]
                    y = slot[3] - 1
                    leng = 0
                    allowed = True
                    while y < self.columns:
                        y += 1
                        while leng < len(words[index]) and allowed == True:
                            if matrix[x][y] == 0:
                                matrix[x][y] = words[index][leng]
                                y += 1
                                leng += 1
                            else:
                                if words[index][leng] != matrix[x][y]:
                                    allowed = False
                                    accepted = False
                                    break
                                else:
                                    matrix[x][y] = words[index][leng]
                                    y += 1
                                    leng += 1
                else:
                    y = slot[3]
                    x = slot[2] - 1
                    leng = 0
                    allowed = True
                    while x < self.lines:
                        x += 1
                        while leng < len(words[index]) and allowed == True:
                            if matrix[x][y] == 0:
                                matrix[x][y] = words[index][leng]
                                x += 1
                                leng += 1                                              
                            else:
                                if words[index][leng] != matrix[x][y]:
                                    x += 1
                                    leng += 1 
                                    accepted = False
                                    allowed = False
                                    break                                                   
                                else:
                                    matrix[x][y] = words[index][leng]
                                    x += 1
                                    leng += 1                                                    
                if accepted == True:
                    nextPossibleMoves.append(slot[0])
                
            return nextPossibleMoves    
    
    def fitness(self):
        return (self.size - len(self.path) + 1)
    
    def calculateDistance(self, next):
        ant = Ant(self.problem)
        ant.path = self.path.copy()
        ant.path.append(next)
        rest = self.size - len(ant.path) + 1
        return (rest - len(ant.nextMoves()))
    
    def addMove(self, traceMatrix, alpha, beta, q0):
        p = [0 for i in range(self.size)]
        nextMoves=self.nextMoves()
        if len(nextMoves) == 0:
            return False
        for i in nextMoves:
            p[i] = self.calculateDistance(i)
        r =[(p[i]**beta)*(traceMatrix[self.path[-1]][i]**alpha) for i in range(len(p))]
        rnd1 = random()
        if rnd1 < q0:
            r = [[i, p[i]] for i in range(len(p))]
            r = max(r, key=lambda a: a[1])
            self.path.append(r[0])
        else:
            s = sum(p)
            if s == 0:
                return choice(nextMoves)
            p = [p[i] / s for i in range(len(p))]
            p = [sum(p[0 : i + 1]) for i in range(len(p))]
            rnd2 = random()
            i = 0
            while rnd2 > p[i]:
                i += 1
            self.path.append(i)
        return True        
    
    
class Controller:
    def __init__(self, problem):
        self.problem = problem;
        self.size = problem.getIndividualSize()
        self.noEpoch = problem.getNoEpoch()
        self.noAnts = problem.getNoAnts()
        self.alpha = problem.getAlpha()
        self.beta = problem.getBeta()
        self.rho = problem.getRho()
        self.q0 = problem.getQ0()
        
    def getSize(self):
        return self.size
    
    def getNoEpoch(self):
        return self.noEpoch
    
    def getNoAnts(self):
        return self.noAnts
    
    def getAlpha(self):
        return self.alpha
    
    def getBeta(self):
        return self.beta
    
    def getRho(self):
        return self.rho
    
    def getQ0(self):
        return self.q0
    
    
    def epoch(self, problem, trace):
        noAnts = problem.getNoAnts()
        size = problem.getIndividualSize()
        alpha = problem.getAlpha()
        beta = problem.getBeta()
        rho = problem.getRho()
        q0 = problem.getQ0()
        
        population = []
        for i in range(noAnts):
            ant = Ant(problem)
            population.append(ant)
        for i in range(size):
            for ant in population:
                ant.addMove(trace, alpha, beta, q0)
        t = [1.0 / population[i].fitness() for i in range(len(population))]
        for i in range(size):
            for j in range(size):
                trace[i][j] = (1 - rho) * trace[i][j]
        for i in range(len(population)):
            for j in range(len(population[i].path) - 1):
                x = population[i].path[j]
                y = population[i].path[j + 1]
                trace[x][y] = trace[x][y] + t[i]
        fitness = [[population[i].fitness(), i] for i in range(len(population))]
        fitness = min(fitness)
        return population[fitness[1]].path
    
   
def main():
    problem = Problem("data01.in", "data02.in", "param01.in")
    controller = Controller(problem);
    size = controller.getSize()
    noEpoch = controller.getNoEpoch()
    noAnts = controller.getNoAnts()
    alpha = controller.getAlpha()
    beta = controller.getBeta()
    rho = controller.getRho()
    q0 = controller.getQ0()
    solution = []
    bestSolution = []
    pheromoneMatrix = [[1 for i in range(size)] for j in range (size)]
    for i in range(noEpoch):
        solution = controller.epoch(problem, pheromoneMatrix)
        if len(solution)>len(bestSolution):
            bestSolution=solution.copy()
    fitness = problem.getIndividualSize() - len(bestSolution)
    print("fitness: ", fitness)
    print("solution: ", bestSolution)
    return fitness

main()