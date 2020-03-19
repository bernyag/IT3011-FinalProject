from tokenize import tokenize

def solve(equation):
    postfixQueue=[]
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
    opStack=[]
    value = 0
    str = [char for char in equation]
    
    for x in str:
        if x == '(':
            opStack.append(x)
        elif x == ')':
            while opStack[-1]!='(':
                value = opStack.pop()
                postfixQueue.append(value)
            opStack.pop()

        elif x =='+' or x =='-' or x=='*' or x =='/':
            while len(opStack)!=0 and opStack[-1]!='(' and (precedenceOf(value)>=precedenceOf(x)):
                value = opStack.pop()                
                postfixQueue.append(value)
            opStack.append(x)
        else:
            postfixQueue.append(x)
    
    while len(opStack)!=0:


        value = opStack.pop()
        postfixQueue.append(value)

    return postfixQueue

def evaluatePostfix(postfixQueue):
    solution=0.0
    operands = []

    while len(postfixQueue)!=0:
        value = postfixQueue.pop(0)
        if not isOperator(value):
            operands.append(value)
        else:
            op1=operands.pop()
            op2=operands.pop()

            result = executeOperation(value,op1,op2)

            operands.append(result)

            #print(postfixQueue.pop(0))
    
    solution = operands.pop()

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
    
    return result



solve("(1+1+1)/3")