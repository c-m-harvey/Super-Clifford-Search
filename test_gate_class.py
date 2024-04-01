from math import prod
import numpy as np
import gate_class
from itertools import product

identity = np.eye(2)
X = np.array([[0, 1],
            [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
            [1j, 0]], dtype=complex)
Z = np.array([[1, 0],
            [0, -1]], dtype=complex)
H = 1/(np.sqrt(2))*np.array([[1, 1],
            [1, -1]])
T = np.array([[1, 0],
            [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def test_quantum_gate():
    good_inputs = ['I','X','Y','Z','T','H'] # bad input caught in runtime test 
    input_dict = dict(zip(good_inputs, [identity,X,Y,Z,T,H]))
    for op_str in good_inputs:
        gate = gate_class.QuantumGate(op_str)
        assert np.allclose(gate.construct_1qubit_gate(), input_dict[op_str]) 
        assert np.allclose(gate.dagger(), np.transpose(np.conj(input_dict[op_str])))
        if op_str == 'T':
            assert not np.allclose(gate.dagger(), gate.construct_1qubit_gate()) # nonhermitian op
        else:
            assert np.allclose(gate.dagger(), gate.construct_1qubit_gate()) # hermitian ops
        assert gate.dim() == (2,2)

def test_nquantum_gate():
    nqb = 2
    good_inputs = ['I','X','Y','Z','T','H'] # bad input caught in runtime test 
    input_dict = dict(zip(good_inputs, [identity,X,Y,Z,T,H]))
    for op_str in good_inputs:
        gate = gate_class.nQuantumGate(op_str, nqb, 1)
        assert np.allclose(gate.construct_n_gate(), np.kron(input_dict[op_str], identity))
        gate = gate_class.nQuantumGate(op_str, nqb, 2)
        assert np.allclose(gate.construct_n_gate(), np.kron(identity, input_dict[op_str]))
        assert gate.dim() == (2**nqb, 2**nqb)
        if op_str == 'T':
            assert not np.allclose(gate.dagger(), gate.construct_n_gate()) # nonhermitian op
        else:
            assert np.allclose(gate.dagger(), gate.construct_n_gate()) # hermitian ops
    for pauli_product in product(list(good_inputs), repeat=2):
        gate = gate_class.nQuantumGate(pauli_product, nqb)
        assert np.allclose(gate.construct_n_gate(), np.kron(input_dict[pauli_product[0]], input_dict[pauli_product[1]]))
        assert gate.dim() == (2**nqb, 2**nqb)
    test_input_arr = np.eye(4)
    gate = gate_class.nQuantumGate('H', 2, 1, array = test_input_arr)
    assert np.allclose(gate.array, test_input_arr)
    gate.construct_n_gate()  # call construction of true array
    assert np.allclose(gate.array, np.kron(H, identity)) # changes as desired

def test_control_gate():
    nqb=2
    good_inputs = ['I','X','Y','Z','T','H'] # bad input caught in runtime test 
    input_dict = dict(zip(good_inputs, [identity,X,Y,Z,T,H]))
    for input in good_inputs:
        C12 = gate_class.ControlGate(input, nqb, cont = 1, targ = 2)
        assert np.allclose(C12.construct_control_gate(), np.block([[identity, np.zeros((2,2))],[np.zeros((2,2)), input_dict[input]]]))
    CX21 = gate_class.ControlGate('Z', nqb, cont = 2, targ = 1)
    check = np.block([[identity, np.zeros((2,2))],[np.zeros((2,2)), Z]])
    assert np.allclose(CX21.construct_control_gate(), check)
    
    CX21 = gate_class.ControlGate('X', nqb, cont = 2, targ = 1)
    check = np.zeros((2**nqb, 2**nqb))
    check[0,0] = 1 # 00
    check[3,1] = 1 # 10
    check[1:3,2:] = X # control = 1 space (10, 11)
    assert np.allclose(CX21.construct_control_gate(), check)

test_quantum_gate()
test_nquantum_gate()
test_control_gate()
