import random
import math
import cmath
from itertools import product
import itertools
from xml.dom.minidom import Identified
import numpy as np
from gate_class import QuantumGate, nQuantumGate, ControlGate
from tqdm import tqdm

def gen_pauli_strings(subspace_str, n):
    assert type(
        subspace_str) == str, "Op_str input should be a string of operators to include, e.g. 'XY' "
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    return [p for p in product(list(subspace_str) , repeat=n)] # return cartesian product (size=n) of provided operators

def gen_pauli_basis(n):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    pauli_space = gen_pauli_strings('XYZI', n)
    return dict((l, nQuantumGate(l, n)) for l in pauli_space)  # returns dictionary {"stringXXX...": matrix,....}
    
def gen_subspace_basis(op_str, n):
    assert type(
        op_str) == str, "Op_str input should be a string of operators to include, e.g. 'XY' "
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    subspace_labels = gen_pauli_strings(op_str, n)
    super_basis = dict((str(np.binary_repr(li, width=n)), nQuantumGate(l, n)) for li, l in enumerate(subspace_labels))
    return super_basis

# Find the subspace representation of trial operator by calculating elements trace( (C_dag * b * C) * b') for basis states {b, b'} in [0,2**n-1]
def calc_super_op(n, trial_op, basis_ops):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    assert type(
        trial_op) == np.ndarray, "op input should be an np.array of dim 2**n x 2**n"
    subspace = gen_subspace_basis(basis_ops, n)
    super_op = np.zeros((2**n, 2**n), dtype=complex)
    column = 0
    for base1 in subspace.values():
        transformed_base = np.transpose(np.conj(trial_op)) @ base1.construct_gate() @ trial_op
        row = 0
        for base2 in subspace.values():
            # find new coefficient of base (1/2**n tr(base' * base))
            coeff = ((1 / 2 ** n) * np.trace(transformed_base @ base2.construct_gate()))
            super_op[row][column] = coeff
            row += 1
        column += 1
    return super_op

# test equality of operators to within approximation error (for approximations utilising result of Solovay-Kitaev)
def approximately_equal(u1, u2, error):
    assert type(
        u1) == np.ndarray, "u1 input should be an np.array of dim 2**n x 2**n"
    assert type(
        u2) == np.ndarray, "u2 input should be an np.array of dim 2**n x 2**n"
    assert u1.shape == u2.shape, "u1 and u2 must have the same shape"
    assert type(error) == float, "error input should be a float value"
    return np.allclose(u1, u2, atol = error)

# Defines 2**n orthonormal basis vectors in Hilbert space
def gen_ONB(n):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    dim = 2 ** n
    basis = dict()
    for d in range(dim):
        base = []
        if d == 0:
            base.append(1)
        else:
            for i in range(d):
                base.append(0)
            base.append(1)
        for j in range(dim - (d + 1)):
            base.append(0)
        basis["base" + str(np.binary_repr(d, width=n))] = np.array(base)
    return basis  # returns dictionary {'base000':[1,0,0,0,...], ....}

def add_hadamards(pool, n):
    assert type(pool) == dict, "Input pool should be a dictionary"
    assert type(
        n) == int, "Input n should be an integer equal to the number of qubits in the system"
    H_pool = dict(("H_"+str(i), nQuantumGate('H', n, targ=i+1)) for i in range(n))
    return pool | H_pool
                                              
def add_T_gates(pool, n):
    assert type(pool) == dict, "Input pool should be a dictionary"
    assert type(
        n) == int, "Input n should be an integer equal to the number of qubits in the system"
    T_pool = dict(("T_"+str(i), nQuantumGate('T', n, targ=i+1)) for i in range(n))
    return pool | T_pool

def add_control_gates(pool, n, op_str):
    assert type(
        n) == int, "Input n should be an integer equal to the number of qubits in the system"
    assert type(pool) == dict, "Input pool should be a dictionary"
    assert type(
        op_str) == str and op_str in ['X', 'Y', 'Z'], "input op_str should be string in ['X','Y','Z']"
    cont_pool = dict()
    for i in range(n):
        for j in range(n):
            if i != j:
                cont_pool["C_"+str(i)+str(j)] = ControlGate(op_str, n, cont = i+1, targ = j+1)
    return pool | cont_pool

# CONDITION 1: must conserve commutation relations
def check_preserves_comm(n, op, basis_ops, error):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    assert type(
        op) == np.ndarray, "op input should be an np.array of dim 2**n x 2**n"
    assert type(error) == float, "error input should be a float value"
    basis = gen_subspace_basis(basis_ops, n)
    bool_preserved = True
    for b1 in basis.values():
        b1_arr = b1.construct_gate()
        for b2 in basis.values():
            b2_arr = b2.construct_gate()
            initial_comm = b1_arr @ b2_arr @ b1_arr @ b2_arr
            transformed_comm = (np.transpose(np.conj(op)) @ b1_arr @ op) @ (np.transpose(np.conj(op)) @ b2_arr @ op) @ (
                np.transpose(np.conj(op)) @ b1_arr @ op) @ (np.transpose(np.conj(op)) @ b2_arr @ op)
            if not approximately_equal(initial_comm, transformed_comm, error):
                bool_preserved = False
                break
    return bool_preserved

# CONDITION 2: must preserve subspace
def check_preserves_space(n, op, basis_ops, error):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    assert type(
        op) == np.ndarray, "op input should be an np.array of dim 2**n x 2**n"
    assert type(error) == float, "error input should be a float value"
    basis = gen_subspace_basis(basis_ops, n)
    bool_clifford = True
    for test in basis.values():
        test_arr = test.construct_gate()
        out = np.transpose(np.conj(op)) @ test_arr @ op
        basis_sum = sum(((1 / 2 ** n) * np.trace(out @ base.construct_gate()))*base.construct_gate() for base in basis.values())
        # if not fully repesented in subspace basis
        if not approximately_equal(basis_sum, out, error):
            bool_clifford = False
        # or if just introduces overall phase to all states
        if approximately_equal(test_arr, -1 * out, error) or approximately_equal(test_arr, -1j * out, error) or approximately_equal(test_arr, 1j * out, error):
            bool_clifford = False
    if check_preserves_comm(n, op, basis_ops, error) == False:
        bool_clifford = False
    return bool_clifford

# CONDITION 3: must map super-Paulis to super-Paulis
def check_takes_pauli_string_to_pauli_string(n, op, basis_ops,  error, return_transforms=False):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    assert type(
        op) == np.ndarray, "op input should be an np.array of dim 2**n x 2**n"
    assert type(error) == float, "error input should be a float value"
    
    s_op = calc_super_op(n, op, basis_ops)
    pauli_strings = gen_pauli_basis(n)

    xs = dict(('X'+str(i), nQuantumGate('X', n, targ=i)) for i in range(n))
    zs =  dict(('Z'+str(i), nQuantumGate('Z', n, targ=i)) for i in range(n))

    clifford = True

    transforms = dict()
    
    for lx, x in xs.items():

        x_arr = x.construct_gate()

        out = np.transpose(np.conj(s_op)) @ x_arr @ s_op # C_dag X C

        found = False

        for kp, p in pauli_strings.items():

            p_arr = p.construct_gate()
            
            if np.isclose((1 / 2**n) * np.trace(p_arr @ out), 1, error) or np.isclose((1 / 2**n) * np.trace(p_arr @ out), -1, error):
                
                transforms.update({lx: kp})
                found = True
                break  # stop checking once found transformed pauli string
        
        if found == False:
            clifford = False
            break

    if clifford is True:

        for lz, z in zs.items():

            z_arr = z.construct_gate()

            out = np.transpose(np.conj(s_op)) @ z_arr @ s_op # C_dag X C
            
            found = False

            for kp, p in pauli_strings.items():
                
                p_arr = p.construct_gate()

                if np.isclose((1 / 2**n) * np.trace(p_arr @ out), 1, error) or np.isclose((1 / 2**n) * np.trace(p_arr @ out), -1, error):
                    
                    transforms.update({lz: kp})
                    found = True
                    break  # stop checking once found transformed pauli string
            
            if found == False:
                clifford = False
                break
    
    if clifford is True:
        if return_transforms is True:
            print("Pauli transforms: ", transforms)
        return True
    else:
        return False

# return True iff all conditions satisfied
def check_clifford(n, op, op_basis, error):
    assert type(
        n) == int, "n input should the an integers equal to the number of qubits in the system"
    assert type(
        op) == np.ndarray, "op input should be an np.array of dim 2**n x 2**n"
    assert type(error) == float, "error input should be a float value"
    if check_preserves_space(n, op, op_basis, error) == True and check_preserves_comm(n, op, op_basis, error) == True and check_takes_pauli_string_to_pauli_string(n, op, op_basis, error) == True:
        return True
    else:
        return False

# Form all possible unitaries from circuit depth restricted universal set
def trial_op_strings(pool, max_depth):
    return [p for p in product(list(pool.keys()) , repeat=max_depth)] # return cartesian product (size=n) of provided operators

def trial_ops(n, pool, max_depth):
    
    trial_circuit_string = trial_op_strings(pool, max_depth)
    
    trial_circuits = []
    for circuit_string in tqdm(trial_circuit_string):
        temp_circuit = np.eye(2**n)
        for op_desc in circuit_string:
            if isinstance(pool[op_desc], ControlGate):
                temp_circuit = temp_circuit @ pool[op_desc].construct_gate()
            else:
                temp_circuit = temp_circuit @ pool[op_desc].construct_gate()
        trial_circuits.append(temp_circuit)

    return dict(zip(trial_circuit_string, trial_circuits))

n = 3
op_basis = 'XY'
approx_pool = dict()
approx_pool = add_hadamards(approx_pool, n)
approx_pool = add_control_gates(approx_pool, n, 'X')
approx_pool = add_T_gates(approx_pool, n)
error = 1e-1
max_gates = np.int(np.ceil(np.log(1 / error)))
print("Max circuit depth: ", max_gates)

trial_dict = trial_ops(n, approx_pool, max_gates)
test_clifford_ops = dict()

for o, (key, op) in tqdm(enumerate(trial_dict.items())):
    if check_clifford(n, op, op_basis, error) == True:
        test_clifford_ops.update({tuple(key): op})
test_clifford_ops.keys()
print("Found: {} Super-Cliffords / {} Super-Operators".format(len(test_clifford_ops), len(trial_dict)))
