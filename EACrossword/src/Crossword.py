'''
Created on 31 mar. 2018

@author: Sara Boanca
'''
import matplotlib.pyplot as plt
import numpy as np


from random import randint, random, shuffle
from copy import deepcopy

class Problem:
    def __init__(self, fileName, fileNameWords):
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
            lin = int(fd.readline())#p
            col = int(fd.readline())#q
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

class Individual:
    def __init__(self, problem):
        self.__problem = problem
        self.__size = problem.getIndividualSize()
        permutation = []
        permutation = [x for x in range(self.__size)]
        shuffle(permutation)
        self.__permutation = permutation
        self.fitness()
        
    def getProblem(self):
        return self.__problem
    
    def getSize(self):
        return self.__size
    
    def getPermutation(self):
        return self.__permutation
    
    def toString(self):
        return str(self.__permutation)
    
    def fitness(self):
        lin = self.__problem.getLines()
        col = self.__problem.getColumns()
        slots = self.__problem.getSlots()
        words = self.__problem.getWords()
        matrix = deepcopy(self.__problem.getMatrix())
        sum = 0
        count = 0
        for i in self.__permutation:
            for slot in slots:
                if slot[0] == i:
                    if slot[1] == 0:
                        for x in range(lin):
                            if x == slot[2]:
                                y = -1
                                while y < col:
                                    y += 1
                                    if y == slot[3]:
                                        length = 0
                                        while length < len(words[count]):
                                            if matrix[x][y] == 0:
                                                matrix[x][y] = words[count][length]
                                                y += 1
                                                length += 1
                                            elif matrix[x][y] == "#":
                                                sum += 2 * (len(words[count]) - slot[4])
                                                break
                                            else:
                                                if words[count][length] != matrix[x][y]:
                                                    sum += 1
                                                    matrix[x][y] = words[count][length]
                                                    y += 1
                                                    length += 1
                                                else:
                                                    matrix[x][y] = words[count][length]
                                                    y += 1
                                                    length += 1 
                                            if y == col and length < len(words[count]):
                                                sum += 2 * (len(words[count]) - length)#removed-1 MULTIPLIED WITH 2
                                                break
                    else:
                        for y in range(col):
                            if y == slot[3]:
                                x = -1
                                while x < lin:
                                    x += 1
                                    if x == slot[2]:
                                        length = 0
                                        while length < len(words[count]):
                                            if matrix[x][y] == 0:
                                                matrix[x][y] = words[count][length]
                                                x += 1
                                                length += 1                                              
                                            elif matrix[x][y] == "#":
                                                sum += 2 * (len(words[count]) - slot[4])                                               
                                                break
                                            else:
                                                if words[count][length] != matrix[x][y]:
                                                    sum += 1
                                                    matrix[x][y] = words[count][length]
                                                    x += 1
                                                    length += 1                                                     
                                                else:
                                                    matrix[x][y] = words[count][length]
                                                    x += 1
                                                    length += 1                                                    
                                            if x == lin and length < len(words[count]):
                                                sum += 2 * (len(words[count]) - length)
                                                break
            count += 1
            
        self.__fitness = sum
    
    def setPermutation(self, perm):
        self.__permutation = perm
        self.fitness()
    
    def setFitness(self, fitness):
        self.__fitness = fitness
        
    def getFitness(self):
        return self.__fitness
    
    def mutation(self, probability, rnd):
        if probability > rnd:
            valMax = len(self.__permutation)
            poz1 = randint(1,valMax - 3)
            poz2 = randint(2,valMax - 2)
            while poz1 >= poz2:
                poz1 = randint(1,valMax - 3)
                poz2 = randint(2,valMax - 2)
            self.__permutation[poz1], self.__permutation[poz2] = self.__permutation[poz2], self.__permutation[poz1]
        return self.__permutation

    def crossover1(self, parent1, parent2, poz1, poz2):
        maxValue = self.__size
        child = [-1 for x in range(maxValue)]
        for i in range(poz1, poz2 + 1):
            child[i] = parent1[i]
        count = i + 1
        j = i + 1
        while count < maxValue:
            if parent2[count] not in child:
                child[j] = parent2[count]
                j += 1
            count += 1
        if count == j and j == maxValue:
            j = 0
            count = 0
            while j < poz1:
                if parent2[count] not in child:
                    child[j] = parent2[count]
                    j += 1
                count += 1
        else:
            count = 0
            while j < maxValue:
                if parent2[count] not in child:
                    child[j] = parent2[count]
                    j += 1
                count += 1
        a = -1
        if a in child:
            j = 0
            while j < poz1:
                if parent2[count] not in child:
                    child[j] = parent2[count]
                    j += 1
                count += 1
        return child 


class Population:
    def __init__(self, dimension, problem):
        self.__dimension = dimension
        self.__problem = problem
        self.__population = [Individual(problem) for _ in range(self.__dimension)]
        
    def getIndividuals(self):
        return self.__population
    
    def getProblem(self):
        return self.__problem
    
    def getDimension(self):
        return self.__dimension
    
    def setDimension(self, dim):
        self.__dimension = dim
    
    def toString(self):
        s = ''
        for i in self.__population:
            s += str(i.getPermutation()) + '\n'
        return s
    
    def evaluate(self):
        for x in self.__population:
            x.fitness()
            
    def reunion(self, newPopulation):
        self.__dimension = self.__dimension + newPopulation.getDimension()
        self.__population = self.__population + newPopulation.getIndividuals()
    
    def selection(self, treshold):
        if treshold < self.__dimension:
            self.__population = sorted(self.__population, key=lambda Individual: Individual.getFitness())
        self.__population = self.__population[:treshold] 
    
    def best(self, treshold):
        aux = sorted(self.__population, key=lambda Individual: Individual.getFitness())
        return aux[:treshold]
    

class Algorithm:
    def __init__(self, fileName, fileNameWords, fileNameParam):
        self.__problem = Problem(fileName, fileNameWords)
        self.__nbIterations = 0
        self.__popDimension = 0
        self.__probMutation = 0
        self.readFromFile(fileNameParam)
        self.__population = Population(self.__popDimension, self.__problem)
        
    def getPopulation(self):
        return self.__population
        
    def getProblem(self):
        return self.__problem
    
    def getNbIterations(self):
        return self.__nbIterations
    
    def getPopDimension(self):
        return self.__popDimension
    
    def getProbMutation(self):
        return self.__probMutation
        
    def readFromFile(self, fileNameParam):
        list = []
        try:
            fd = open(fileNameParam, 'r')
            self.__nbIterations = int(fd.readline()) 
            self.__popDimension = int(fd.readline())
            self.__probMutation = int(fd.readline())/100
        except IOError:
            print("error")
    
    def iteration(self):
        shuffle(self.__population.getIndividuals())
        valMax = self.__population.getIndividuals()[0].getSize()
        i1 = randint(0, self.__popDimension - 1)
        i2 = randint(0, self.__popDimension - 1)
        if i1 != i2:
            parent1 = self.__population.getIndividuals()[i1]
            parent2 = self.__population.getIndividuals()[i2]
            child1 = Individual(self.__problem)
            poz1 = randint(0,valMax - 2)
            poz2 = randint(1,valMax - 1)
            while poz1 >= poz2:
                poz1 = randint(0,valMax - 2)
                poz2 = randint(1,valMax - 1)
            child1.setPermutation(child1.crossover1(parent1.getPermutation(), parent2.getPermutation(), poz1, poz2))
            rnd = random()
            child1.setPermutation(child1.mutation(0.02, rnd))
            if parent1.getFitness() > parent2.getFitness() and parent1.getFitness() > child1.getFitness():
                self.__population.getIndividuals()[i1] = child1
            if parent2.getFitness() > parent1.getFitness() and parent2.getFitness() > child1.getFitness():
                self.__population.getIndividuals()[i2] = child1
            
    def run(self):
        list=[]
        stdDev=[]
        mean=[]
        for i in range(self.__nbIterations):
            self.iteration()
        a = self.__population.best(1)
        return a[0].getFitness()
        
class Application:
    def __init__(self):
        pass
    
    def main(self):
        algorithm = Algorithm("data03.in", "data04.in", "param02.in")
        return algorithm.run()
    
    def plotRun(self):
        list=[]
        stdDev=[]
        mean=[]
        for i in range(30):
            list.append(self.main())
        arr = np.array(list)
        for i in range(30):
            m = np.mean(arr, axis=0)
            mean.append(m)
            stdev = np.std(arr, axis=0)
            stdDev.append(stdev)
        plt.plot(range(0, 30),mean,"r*")
        plt.plot(range(0, 30),stdDev)
        plt.plot(range(0, 30),list,'go')
        plt.show()  
'''
def tests():
    algorithm = Algorithm("data03.in", "data04.in", "param02.in")
    assert(algorithm.getNbIterations() == 6000)
    assert(algorithm.getPopDimension() == 150)
    assert(algorithm.getProbMutation() == 0.02)
    problem = algorithm.getProblem()
    assert(problem.getLines() == 5)
    assert(problem.getColumns() == 5)
    assert(problem.getBlanks() == 6)
    assert(problem.getWords() == ["sara","are","crin","nia","ril","sc","elan", "acuma", "ea", "ana"])
    assert(problem.getSlots() == [[0,0,0,0,4],[1,0,1,0,4],[2,0,2,1,4],[3,0,4,0,5],[4,1,0,0,2],[5,1,3,0,2],[6,1,0,1,3],[7,1,0,2,3],[8,1,0,3,3],[9,1,2,4,3]])
    assert(problem.getIndividualSize() == 10)
    population = algorithm.getPopulation()
    assert(population.getProblem() == problem)
    assert(population.getDimension() == 100)
    individual = population.getIndividuals()[0]
    assert(individual.getSize() == 10)
    assert(individual.getProblem() == problem)
    individual.setPermutation([0,6,1,9,7,4,2,3,5])
    assert(individual.getFitness() == 0)
    individual.setPermutation([0,6,1,7,9,4,2,3,5])
    assert(individual.getFitness() == 4)
    individual.setPermutation([0,6,1,9,7,4,2,3,5,8])
    assert(individual.getFitness() == 0)
    individual.setPermutation([0,8,1,9,7,4,2,3,5,6])
    assert(individual.getFitness() == 4)
    individual.setFitness(111)
    assert(individual.getFitness() == 111)
    individual.setPermutation([4, 1, 3, 9, 5, 7, 2, 6, 0, 8])
    print(individual.getFitness())
    assert(individual.crossover1([1,8,3,0,9,2,7,4,6,5],[1,2,0,6,7,3,8,5,4,9],5,8) == [1,0,3,8,5,2,7,4,6,9])
    assert(individual.crossover1([1,8,2,4,7,6,3,9,5,0],[9,0,3,5,7,2,8,6,4,1],3,6) == [5,2,8,4,7,6,3,1,9,0])
    assert(individual.crossover1([1,8,2,4,7,6,3,9,5,0],[9,0,3,5,7,2,8,6,4,1],0,4) == [1,8,2,4,7,6,9,0,3,5])
    print("tests passed")


def test():
    algorithm = Algorithm("data03.in", "data04.in", "param02.in")
    problem = algorithm.getProblem()
    population = algorithm.getPopulation()
    individual = population.getIndividuals()[0]
    individual.setPermutation([0,6,1,7,9,4,2,3,5,8])
    assert(individual.getFitness() == 4)
    individual.setPermutation([0,6,1,9,7,4,2,3,5,8])
    assert(individual.getFitness() == 0)
    print("begin")
    individual.setPermutation([4, 6, 9, 3, 8, 0, 5, 1, 2, 7])
    assert(individual.getFitness() == 11)
'''
app = Application()
app.plotRun()
