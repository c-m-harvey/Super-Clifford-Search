import numpy as np

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

class QuantumGate:

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

    def __init__(self, name: str, array=None):
        self.name = name
        if array is not None:
            self.array = array
            self.size = self.dim()
            assert(np.allclose(self.array@self.dagger(), np.eye(self.array.shape[0]))) # assert unitary

    def dim(self):
        assert self.array.shape == (2,2) # this class construct single qubit gates
        return self.array.shape

    def construct_1qubit_gate(self, op_name = None) -> np.array:
        if op_name is None:
            name = self.name
        else:
            name = op_name
        assert name in ['I', 'X', 'Y', 'Z', 'T', 'H'], "Choose from I,X,Y,Z,H,T to return a one qubit gate or define your own Quantum Gate"
        if name == 'I':
            array = self.identity
        elif name == 'X':
            array = self.X
        elif name == 'Y':
            array = self.Y
        elif name == 'Z':
            array = self.Z
        elif name == 'H':
            array = self.H
        elif name == 'T':
            array = self.T
        else:
            NotImplementedError
            return None
        self.array = array
        return array

    def dagger(self) -> np.array:
        return np.transpose(np.conj(self.array))
   
    def __str__(self) -> str:
        return f'Single qubit {self.name} gate called'

class nQuantumGate(QuantumGate):

    def __init__(self, name: str, nqb: int, targ=None, array=None):
        self.nqb = nqb # input space
        super().__init__(name, array)
        if targ is not None:
            self.targ = targ # target qubit
        if array is not None:
            self.size = self.dim() # check input array is consistent with input nqb

    def dim(self) -> np.array:
        cod = 2 ** self.nqb
        assert self.array.shape == (cod,cod), "input array is not (2**n, 2**n) for n qubits"
        return self.array.shape

    def construct_n_gate(self) -> np.array:
        if len(self.name) == 1:
            pauli_string = "".join('I' if i != (self.targ-1) else self.name for i in range(self.nqb))
        else:
            pauli_string = self.name
        for i in range(self.nqb): 
            if i == 0:
                b = super().construct_1qubit_gate(pauli_string[i])
            else: 
                c = super().construct_1qubit_gate(pauli_string[i])
                b = np.kron(b, c)
        self.array = b
        return self.array

    def __str__(self) -> str:
        return f'{self.nqb}-qubit {self.name} gate called acting on qubit {self.targ}'

class ControlGate(QuantumGate):

    def __init__(self, name: str, nqb: int, cont: int, targ: int, array = None):
        self.nqb = nqb
        self.cont = cont
        self.targ = targ
        assert self.cont != self.targ, "control qubit cannot be the same as target qubit"
        super().__init__(name, array)
        if array is not None:
            self.size = self.dim()
    
    def dim(self):
        cod = 2**self.nqb
        assert self.array.shape == (cod,cod), "input array is not (2**n, 2**n) for n qubits"
        return self.array.shape

    def construct_control_gate(self) -> np.array:
        
        cod = 2 ** self.nqb
        bin_rep = [np.binary_repr(i, width=self.nqb) for i in range(cod)]
        bin_states = gen_ONB(self.nqb)
        
        C_ij = np.zeros((cod, cod)) 

        for b in bin_rep:  # basis state representation 

            if b[self.cont-1] == '0':  # if basis state has control = 0 apply I*| b > < b|

                C_ij = np.add(C_ij, np.outer(bin_states["base" + str(b)],
                                            np.transpose(bin_states["base" + str(b)])))

            elif b[self.cont-1] == '1':  # if basis state has control = 1 apply Xj * | b > < b |
                
                op = nQuantumGate(self.name, self.nqb, targ=self.targ).construct_n_gate() 
                transform_basis = op @ bin_states["base" + str(b)]      
                
                C_ij = np.add(C_ij, np.outer(
                    transform_basis, np.transpose(bin_states["base" + str(b)]))) 
        
        self.array = C_ij
        return self.array

    def __str__(self) -> str:
        return f'{self.nqb}-qubit control {self.name} gate called acting on qubit {self.targ} with control qubit {self.cont}'

# note: class ParameterisedGate(QuantumGate) would require **params and .sub function .eval 
