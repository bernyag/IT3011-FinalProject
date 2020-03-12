from queue import Queue
from queue import LifoQueue
from tokenize import tokenize

def solve(equation):
    postfixQueue=Queue()
    postfixQueue=parseInfixToPostfix(equation,postfixQueue)

    solution = evaluatePostfix(postfixQueue)

    print(solution)

    #return solution

def isOperator(character):
    if character=='+'or character=='-'or character=='*'or character=='/':
        return True
    else:
        return False

def parseInfixToPostfix(equation,postfixQueue):
    opStack=LifoQueue()
    value = 0
    str = [char for char in equation]
    
    for x in str:
        print(x)
        if x == '(':
            opStack.put(x)
        elif x == ')':
            value=opStack.get()
            while value!='(':
                postfixQueue.put(value)
                value = opStack.get()
        elif x =='+' or x =='-' or x=='*' or x =='/':
            if not opStack.empty():
                value = opStack.get()
            while not opStack.empty() and value!='(' and (precedenceOf(value)>=precedenceOf(x)):
                postfixQueue.put(value)
                value = opStack.get()
            opStack.put(value)
            opStack.put(x)
        else:
            postfixQueue.put(x)
    
    while not opStack.empty():


        value = opStack.get()
        postfixQueue.put(value)

    return postfixQueue

def evaluatePostfix(postfixQueue):
    solution=0.0
    operands = LifoQueue()

    while not postfixQueue.empty():
        value = postfixQueue.get()
        if not isOperator(value):
            operands.put(value)
        elif postfixQueue.empty():
            print(here)
            break
        else:
            op1=operands.get()
            op2=operands.get()

            result = executeOperation(value,op1,op2)
            print(result)
            operands.put(result)
    
    solution = operands.get()
    print(solution)
    return solution

def precedenceOf(theOperator):
    precedence = 0

    if theOperator == '+' or theOperator=='-':
        precedence=1
    elif theOperator=='*' or theOperator=='/':
        precedence=2

    return precedence

def executeOperation(op,op1,op2):
    result = 0.0

    op1=int(op1)
    op2=int(op2)

    if op == '+':
        result=op2+op1
    elif op =='-':
        result=op2-op1
    elif op =='*':
        result=op2*op1
    elif op == '/':
        result=op2/op1

    print(result)
    
    return result



solve("1+1")