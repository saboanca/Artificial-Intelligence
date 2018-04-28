'''
Created on 15 mar. 2018

@author: Sara Boanca
'''

"""
NxN Puzzle Game:

    For a given puzzle of n x n squares with numbers from 1 to (n x n-1) (one square is
    empty) in an initial configuration, find a sequence of movements for the numbers in order to
    reach a final given configuration, knowing that a number can move (horizontally or vertically) on
    an adjacent empty square. In Figure 7 are presented two examples of puzzles (with the initial
    and final configuration).
    
    BFS and GBFS
"""

from time import time
from math import *

class Configuration:
    '''
    holds a configuration of pieces
    '''
    def __init__(self, positions):
        self.__size = len(positions)
        self.__values = positions[:]
    
    def getSize(self):
        return self.__size
    
    def getValues(self):
        return self.__values[:]

    def nextConfig(self, poz, n):
        '''
        Move the piece from position poz in the next valid position
        input params: poz - the position of the piece to be moved
                      n   - the input size of the nxn puzzle
        output params: the list of the next valid configuration
        '''
        nextConfig = []
        
        if poz + n < self.__size:
            nextStep = self.__values[:]
            nextStep[poz] = self.__values[poz + n]
            nextStep[poz + n] = 0
            nextConfig.append(Configuration(nextStep))
            #move 0 piece downwards
            
        if poz - n >= 0:
            nextStep = self.__values[:]
            nextStep[poz] = self.__values[poz - n]
            nextStep[poz - n] = 0
            nextConfig.append(Configuration(nextStep))
            #move 0 piece upwards
        
        if poz % n != 0:
            nextStep = self.__values[:]
            nextStep[poz] = nextStep[poz - 1]
            nextStep[poz - 1] = 0
            nextConfig.append(Configuration(nextStep))

        if poz % n < n - 1:
            nextStep = self.__values[:]
            nextStep[poz] = nextStep[poz + 1]
            nextStep[poz + 1] = 0
            nextConfig.append(Configuration(nextStep))
                  
        return nextConfig
        
    def __eq__(self, config):
        if not isinstance(config, Configuration):
            return False
        if self.__size != config.getSize():
            return False
        for i in range(self.__size):
            if self.__values[i] != config.getValues()[i]:
                return False
        return True
    
    def __str__(self):
        n = int(sqrt(self.getSize()))
        cn = n*n
        string = ''
        j = 1
        for i in range(cn):
            string += " " + str(self.__values[i])
            if (i + 1) % n == 0:
                string += "\n"
        return string   
    
class State:
    '''
    holds a path of configurations
    '''
    
    def __init__(self):
        self.__values = []
    
    def setValues(self, values):
        self.__values = values[:]
        
    def getValues(self):
        return self.__values[:]
    
    def __str__(self, ):
        ret = ""
        for i in self.__values:
            ret += str(i) + "\n"
        return ret
    
    def __add__(self, someClass):
        state = State()
        if isinstance(someClass, State):
            state.setValues(self.__values + someClass.getValues())
        elif isinstance(someClass, Configuration):
            state.setValues(self.__values + [someClass])
        else:
            state.setValues(self.__values)
        return state
    
class Problem:
    def __init__(self, fileName):
        self.__n = None
        self.__initialConfig = None
        self.__finalConfig = None
        self.__fileName = fileName
        if not self.readFromFile(fileName):
            print("Error reading from file")
            exit(1)
        self.__initialState = State()
        self.__initialState.setValues([self.__initialConfig])
        
    def readFromFile(self, fileName):
        try:
            fd = open(fileName, 'r')
            self.setN(int(fd.readline()))
            list1 = ([int(x) for x in fd.readline().split(' ')])
            list2 = []
            for i in list1:
                if i != 0:
                    list2.append(str(i))
                else:
                    list2.append(i)
            self.setInitial(Configuration(list2))
            list3 = ([int(x) for x in fd.readline().split(' ')])
            list4 = []
            for i in list3:
                if i != 0:
                    list4.append(str(i))
                else:
                    list4.append(i)
            self.setFinal(Configuration(list4))
        except IOError:
            return False
        return True
      
        
    def expand(self, currentState, n):
        list = []
        currentConfig = currentState.getValues()[-1]
        for i in range(currentConfig.getSize()):
            if currentConfig.getValues()[i] == 0:
                for j in currentConfig.nextConfig(i, n):
                    list.append(currentState + j)
        return list
    
    def setFinal(self, finalConfig):
        self.__finalConfig = finalConfig
        
    def setInitial(self, initialConfig):
        self.__initialConfig = initialConfig
        
    def setN(self, n):
        self.__n = n
        
    def getInitial(self):
        return self.__initialConfig
    
    def getFinal(self):
        return self.__finalConfig
    
    def getRoot(self):
        return self.__initialState
    
    def getN(self):
        return self.__n
    
    def heuristics(self, state, finalConfig):
        len = finalConfig.getSize()
        count = 0
        for i in range(len):
            if state.getValues()[-1].getValues()[i] != finalConfig.getValues()[i]:
                count -= 1
        return count
    
class Controller:
    def __init__(self, problem):
        self.__problem = problem
        
    def BFS(self, root, n):
        #node = currentState
        toVisit = [root]
        visited = {}
        while len(toVisit) > 0:
            node = toVisit.pop(0)
            visited[str(node.getValues()[-1])] = "visited"
            if node.getValues()[-1] == self.__problem.getFinal():
                return node
            else:
                children = self.__problem.expand(node, n)
                for i in children:
                    if str(i.getValues()[-1]) not in visited:
                        toVisit = toVisit + [i]
                    
    
    def GBFS(self, elem, n):
        start = self.__problem.getRoot()
        visited = {}
        toVisit = [start]
        while len(toVisit) != 0:
            node = toVisit.pop(0)
            print(node)
            visited[str(node.getValues()[-1])] = "visited"
            if node.getValues()[-1] == elem:
                return node
            else:
                children = self.__problem.expand(node, n)
                aux = []
                for i in children:
                    if str(i.getValues()[-1]) not in visited:
                        aux = aux + [i]
                aux = [[i, self.__problem.heuristics(i, self.__problem.getFinal())] for i in aux]
                aux.sort(key=lambda i:i[1])
                aux = [i[0] for i in aux]
                toVisit = toVisit + aux
   
class UI:
    def __init__(self):
        self.__problem = Problem("3pdf.txt")
        self.__controller = Controller(self.__problem)
        
    def printMenu(self):
        str = ""
        str += "\t MENU\n\n"
        str += "1. BFS Method\n"
        str += "2. GBFS Method\n"
        str += "0. Exit\n"
        print(str) 
        
    def findPathBFS(self):
        startTime = time()
        print(str(self.__controller.BFS(self.__problem.getRoot(), self.__problem.getN())))
        print('execution time: ', time() - startTime, 'seconds')
        
    def findPathGBFS(self):
        startTime = time()
        print(str(self.__controller.GBFS(self.__problem.getFinal(), self.__problem.getN())))
        print('execution time: ', time() - startTime, 'seconds')
        
    def run(self):
        keepAlive = True
        while keepAlive == True:
            self.printMenu()
            #try:
            command = int(input("command: "))
            if command == 0:
                keepAlive = False
                break
            elif command == 1:
                self.findPathBFS()
            elif command == 2:
                self.findPathGBFS()
            #except:
            #    print("Invalid command! Try again!\n")
        

def main():
    ui = UI()
    ui.run()
main()