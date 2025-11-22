# Python program for implementation
# of Ford Fulkerson algorithm
import sys
from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_squared_error
from numpy import inf

purchase_fee = 0.2
compensation_fee = 1
cap_list = [100, 150, 200, 250]
testmarkNum = 300
itemNum = 10
#startmark = int(sys.argv[1])
startmark = 0
endmark = startmark + 10

def actual_obj(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    selectedNum_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        cnt = num * itemNum
        for i in range(itemNum):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(itemNum):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        sol = []
        selectedItemNum = 0
        for i in range(itemNum):
            sol.append(x[i].x)
            if x[i].x == 1:
              selectedItemNum = selectedItemNum + 1
        objective = m.objVal
        obj_list.append(objective)
        selectedNum_list.append(selectedItemNum)
        # print(selectedItemNum)
#        print("TOV: ", sol, objective)
        
    return np.array(obj_list), np.array(selectedNum_list)


def correction_single_obj(realPrice, predPrice, cap, realWeightTemp, predWeightTemp):
#    print("realPrice: ", realPrice)
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        predSol = []
        x1_selectedItemNum = 0
        for i in range(itemNum):
            predSol.append(x[i].x)
            if x[i].x == 1:
              x1_selectedItemNum = x1_selectedItemNum + 1
        objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)

        # Stage 2:
        realWeight = realWeight.tolist()
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x = m2.addVars(itemNum, vtype=GRB.BINARY, name='x')
        sigma = m2.addVars(itemNum, vtype=GRB.BINARY, name='sigma')

        OBJ = purchase_fee * x.prod(realPrice)
        for i in range(itemNum):
            OBJ = OBJ - compensation_fee * realPrice[i] * sigma[i]
        m2.setObjective(OBJ, GRB.MAXIMIZE)

        m2.addConstr((x.prod(realWeight) - sigma.prod(realWeight)) <= cap)
        for i in range(itemNum):
            m2.addConstr(x[i] == predSol[i])
            m2.addConstr(x[i] >= sigma[i])

        m2.optimize()
        objective = m2.objVal
        sol = []
        x2_selectedItemNum = 0
        for i in range(itemNum):
            sol.append(x[i].x - sigma[i].x)
            if x[i].x - sigma[i].x == 1:
              x2_selectedItemNum = x2_selectedItemNum + 1
#        print("Stage 2: ", sol, objective)

    return objective, x1_selectedItemNum, x2_selectedItemNum
    
    
def compute_single_correction_obj(predSol, realPrice, realWeightTemp, cap):
#    print("realPrice: ", realPrice)
    realWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
    
    # Stage 2:
    realWeight = realWeight.tolist()
    m2 = gp.Model()
    m2.setParam('OutputFlag', 0)
    x = m2.addVars(itemNum, vtype=GRB.BINARY, name='x')
    sigma = m2.addVars(itemNum, vtype=GRB.BINARY, name='sigma')

    OBJ = purchase_fee * x.prod(realPrice)
    for i in range(itemNum):
        OBJ = OBJ - compensation_fee * realPrice[i] * sigma[i]
    m2.setObjective(OBJ, GRB.MAXIMIZE)

    m2.addConstr((x.prod(realWeight) - sigma.prod(realWeight)) <= cap)
    for i in range(itemNum):
        m2.addConstr(x[i] == predSol[i])
        m2.addConstr(x[i] >= sigma[i])

    m2.optimize()
    objective = m2.objVal
    sol = []
    x2_selectedItemNum = 0
    for i in range(itemNum):
        sol.append(x[i].x - sigma[i].x)
        if x[i].x - sigma[i].x == 1:
          x2_selectedItemNum = x2_selectedItemNum + 1
#        print("Stage 2: ", sol, objective)

    return objective


def check_feasible(realPrice, predPrice, cap, realWeightTemp, predWeightTemp):
#    print("realPrice: ", realPrice)
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    
    feaOrNot = True
#    print(realWeightTemp)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        predSol = np.zeros(itemNum)
        for i in range(itemNum):
            predSol[i] = x[i].x
        objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)
    totalWeight = np.dot(predSol, realWeight)
#    print(predSol, totalWeight)
    if totalWeight <= cap:
        feaOrNot = True
    else:
        feaOrNot = False

    return feaOrNot


#print('CombOptNet')
#for penalty_term in penaltyList:
print('compensation_fee: ', compensation_fee)
for capacity in cap_list:
    print(capacity)
    for testmark in range(startmark, endmark):
    #    capacity = capacity + 50
    #    print(capacity)
    #for testmark in range(1):
        priceT = np.loadtxt('./CombOptNet/trainSize700/proposed_prices' + str(capacity) +'/proposed_prices' + str(compensation_fee) +'(' + str(testmark) + ').txt')
        weightT = np.loadtxt('./CombOptNet/trainSize700/proposed_weights' + str(capacity) +'/proposed_weights' + str(compensation_fee) +'(' + str(testmark) + ').txt')
    #        penaltyTemp = np.loadtxt('./CombOptNet/trainSize700/test_penalty' + str(penalty_term) +'/test_penalty(' + str(testmark) + ').txt')
    #    weightT = np.loadtxt('./CombOptNet/rf_weights.txt')
        realPriceT = priceT[:, 0]
        predPriceT = priceT[:, 1]
        realWeightT = weightT[:, 0]
        predWeightT = weightT[:, 1]
        realPriceWeight = np.vstack((realPriceT, realWeightT))
        predPriceWeight = np.vstack((predPriceT, predWeightT))
        meanPriceVal = np.mean(realPriceT)
        meanWeightVal = np.mean(realWeightT)

        real_obj, real_selected_num = actual_obj(realPriceT, capacity, realWeightT, n_instance=testmarkNum)
    #    print(np.mean(real_obj))
        corr_obj_list = []
        pred_sol_temp = np.loadtxt('./' + str(compensation_fee) + '_700/CombOptNet_cap' + str(capacity) + '_sols(' + str(testmark) + ').txt')
#        print(pred_sol_temp)
#        pred_sol_temp = np.loadtxt('CombOptNet_0.21_sols.txt')
        x1_selected_num_list = []
        x2_selected_num_list = []
        feasibleNum = 0
        for testNum in range(testmarkNum):
        #        print(testNum)
           realWT = {}
           predWT = {}
           realValue = {}
           predValue = {}
           predSol = np.zeros(itemNum)
    #           penalty = np.zeros((itemNum))
           for i in range(itemNum):
               realWT[i] = realWeightT[i+testNum*itemNum]
#               predWT[i] = predWeightT[i+testNum*itemNum]
               predWT[i] = 1
               realValue[i] = realPriceT[i+testNum*itemNum]
#               predValue[i] = predPriceT[i+testNum*itemNum]
               predValue[i] = realPriceT[i+testNum*itemNum]
               predSol = pred_sol_temp[testNum]
    #               penalty[i] = penaltyTemp[i+testNum*itemNum]

    #           feaOrNot = check_feasible(realValue, predValue, capacity, realWT, predWT, penalty)
    #           if feaOrNot:
    #                feasibleNum = feasibleNum + 1
#           print(realWT,predWT)
           corrrlst = compute_single_correction_obj(predSol, realValue, realWT, capacity)
           corr_obj_list.append(corrrlst)
    #           x1_selected_num_list.append(x1_selected_num)
    #           x2_selected_num_list.append(x2_selected_num)
        #        print(corrrlst)

        print("MSE: ", mean_squared_error(realPriceWeight, predPriceWeight), "avgCorrReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, "avgTOV: ", sum(real_obj)/testmarkNum)

    print("\n")
    #    print("MSE: ", mean_squared_error(realPriceWeight, predPriceWeight), "feasibleRatio: ", feasibleNum/testmarkNum)
    #    print("MSE: ", mean_squared_error(realPriceWeight, predPriceWeight), "avgCorrReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, "avgTOV: ", sum(real_obj)/testmarkNum, "TOV_avgSelectedNum: ", sum(real_selected_num)/testmarkNum, "x1_avgSelectedNum: ", sum(x1_selected_num_list)/testmarkNum, "x2_avgSelectedNum: ", sum(x2_selected_num_list)/testmarkNum)
    #        print(mean_squared_error(realPriceWeight, predPriceWeight), ",", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, ",", sum(real_obj)/testmarkNum, ",", sum(real_selected_num)/testmarkNum, ",", sum(x1_selected_num_list)/testmarkNum, ",", sum(x2_selected_num_list)/testmarkNum)
    #    print("TOV: ", sum(real_obj))
    #    print("TOV: ", sum(real_obj))


## check feasibility
#print('2S')
##for penalty_term in penaltyList:
##    print('Penalty: ', penalty_term)
#capacity = 250
#compensation_fee_list = [0.21, 0.25, 0.3, 0.4, 0.6, 1]
#for compensation_fee in compensation_fee_list:
#    print(compensation_fee)
##for capacity in capacity_list:
##    print(capacity)
#    for testmark in range(startmark, endmark):
#        priceT = np.loadtxt('./CombOptNet/trainSize700/proposed_prices' + str(capacity) +'/proposed_prices' + str(compensation_fee) +'(' + str(testmark) + ').txt')
#        weightT = np.loadtxt('./CombOptNet/trainSize700/proposed_weights' + str(capacity) +'/proposed_weights' + str(compensation_fee) +'(' + str(testmark) + ').txt')
#    #        penaltyTemp = np.loadtxt('./CombOptNet/trainSize700/test_penalty' + str(penalty_term) +'/test_penalty(' + str(testmark) + ').txt')
#    #    weightT = np.loadtxt('./CombOptNet/rf_weights.txt')
#        realPriceT = priceT[:, 0]
#        predPriceT = priceT[:, 1]
#        realWeightT = weightT[:, 0]
#        predWeightT = weightT[:, 1]
#        realPriceWeight = np.vstack((realPriceT, realWeightT))
#        predPriceWeight = np.vstack((predPriceT, predWeightT))
#        meanPriceVal = np.mean(realPriceT)
#        meanWeightVal = np.mean(realWeightT)
#
#        feasibleNum = 0
#        corr_obj_list = []
#        x1_selected_num_list = []
#        x2_selected_num_list = []
#        feasibleNum = 0
#        for testNum in range(testmarkNum):
#        #        print(testNum)
#           realWT = {}
#           predWT = {}
#           realValue = {}
#           predValue = {}
#    #           penalty = np.zeros((itemNum))
#           for i in range(itemNum):
#               realWT[i] = realWeightT[i+testNum*itemNum]
#               predWT[i] = predWeightT[i+testNum*itemNum]
#               realValue[i] = realPriceT[i+testNum*itemNum]
#               predValue[i] = predPriceT[i+testNum*itemNum]
#    #               penalty[i] = penaltyTemp[i+testNum*itemNum]
##               print(realWT,predWT)
#           feaOrNot = check_feasible(realValue, predValue, capacity, realWT, predWT)
#           if feaOrNot:
#                feasibleNum = feasibleNum + 1
#
#        print("MSE: ", mean_squared_error(realPriceWeight, predPriceWeight), "feasibleRatio: ", feasibleNum/testmarkNum)
#
#    print("\n")
