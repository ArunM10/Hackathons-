import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
import time

def HVA_layer(params, Delta, wires):
    
    qml.MultiRZ(params[0]*(-1*Delta),wires=(int(wires[0]), int(wires[len(wires)-1])))
    qml.IsingXX(params[1]*(-1),wires=(int(wires[0]), int(wires[len(wires)-1])))
    qml.IsingYY(params[1]*(-1),wires=(int(wires[0]), int(wires[len(wires)-1])))
    for i in range(int(len(wires)/2)-1):
        qml.MultiRZ(params[0]*(-1*Delta),wires=(int(wires[2*i+1]), int(wires[2*i+2])))
        qml.IsingXX(params[1]*(-1),wires=(int(wires[2*i+1]), int(wires[2*i+2])))
        qml.IsingYY(params[1]*(-1),wires=(int(wires[2*i+1]), int(wires[2*i+2])))
    for i in range(int(len(wires)/2)):
        qml.MultiRZ(params[2]*(-1*Delta),wires=(int(wires[2*i]), int(wires[2*i+1])))
        qml.IsingXX(params[3]*(-1),wires=(int(wires[2*i]), int(wires[2*i+1])))
        qml.IsingYY(params[3]*(-1),wires=(int(wires[2*i]), int(wires[2*i+1])))

def HVA_ansatz(params, Delta, layers, wires):
    
    # We first prepare the initial state;
    # We follow Wiersema et. al and choose
    # the inital state to be the product of 
    # bell states. 
    
    [qml.PauliX(wires=int(wires[i])) for i in range(len(wires))]
    [qml.Hadamard(wires=int(wires[2*i])) for i in range(int(len(wires)/2))]
    [qml.CNOT(wires=[int(wires[2*i]),int(wires[2*i+1])]) for i in range(int(len(wires)/2))]
    
    # Now we add HVA_layers:
    
    for i in range(layers):
        HVA_layer(params[4*i:4*(i+1)], Delta, wires)
        

def H_XXZ(Delta, wires):
    
    # We now prepare the Hamiltonian of the XXZ model.
    # H = \sum_i (X_{i}X_{i+1} + Y_{i}Y_{i+1} + \Delta Z_{i}Z_{i+1} ) 
    
    coeffs = []
    observables = []
    
    for i in range(len(wires)-1):
        
        observables.append(qml.PauliX(wires[i])@qml.PauliX(wires[i+1]))
        observables.append(qml.PauliY(wires[i])@qml.PauliY(wires[i+1]))
        observables.append(qml.PauliZ(wires[i])@qml.PauliZ(wires[i+1]))
        coeffs.append(1)
        coeffs.append(1)
        coeffs.append(Delta)
        
    
    observables.append(qml.PauliX(wires[-1])@qml.PauliX(wires[0]))
    observables.append(qml.PauliY(wires[-1])@qml.PauliY(wires[0]))
    observables.append(qml.PauliZ(wires[-1])@qml.PauliZ(wires[0]))
    coeffs.append(1)
    coeffs.append(1)
    coeffs.append(Delta)
    
    return qml.Hamiltonian(coeffs, observables)


def corr_function(i, params, Delta, layers, total_qubits, perturbation = None, timesteps = 0):
    
    dev1 = qml.device("default.qubit", wires= total_qubits) 
    
    @qml.qnode(dev1)
    def circuit(i, params, Delta, layers, total_qubits, perturbation = None, timesteps = 0):
    
        HVA_ansatz(params, Delta, layers, wires = range(total_qubits))
        
        if perturbation == 'state':
            
            [qml.Hadamard(wires=int(k)) for k in range(total_qubits)]
        
        elif perturbation == 'hamiltonian':
            
            Delta = 0.5
        
        for _ in range(timesteps):
            
            dt = 1/400
            pars = dt*np.ones(4)
            HVA_layer(pars, Delta, wires = range(total_qubits))
            
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(int(i)))
    
    return circuit(i, params, Delta, layers, total_qubits, perturbation, timesteps)

def entanglement_entropy(params, qubits_A, Delta, layers, total_qubits, perturbation = None, timesteps = 0):
    
    dev1 = qml.device("default.qubit", wires = total_qubits)
    @qml.qnode(dev1)
    def density_matrix(params, qubits_A, Delta, layers, total_qubits):
        
        HVA_ansatz(params, Delta, layers, wires = range(total_qubits))
        
        if perturbation:
            [qml.Hadamard(wires=k) for k in range(total_qubits)]
        
        for _ in range(timesteps):
            dt = len(qubits_A)/400
            pars = dt*np.ones(4)
            HVA_layer(pars, Delta, wires = range(total_qubits))
            
        
        return qml.density_matrix(qubits_A)
    
    rho_A = density_matrix(params,qubits_A, Delta, layers, total_qubits)
    
    ev = np.linalg.eigvals(rho_A)
    
    ent = 0
    
    for ev1 in ev:
        if ev1 > 0:
            ent = ent - ev1*np.log(ev1)
            
    return ent    

