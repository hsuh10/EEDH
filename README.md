# EEDH: Microscopic Rate Constant Calculator
This is a Python-based tool for calculating **RRKM (Rice–Ramsperger–Kassel–Marcus) unimolecular rate constants** with optional tunneling corrections. Density of states counting is achieved by counting the full quantum states of reactants and transition state molecules one by one in terms of vibrational and rotational degrees of freedom based on dynamic programming. It is only used to calculate the microscopic rate constant k(E) of a single path, and the current calculation speed is relatively slow.

To use this code, you need to install the relevant libraries first:
```bash
pip install numpy scipy matplotlib
```

example.txt serves as the input file. The frequency lists of reactants and TS are required. Among them, the TS frequency list must exclude imaginary frequency modes and is also a required item. The energy barrier and reverse energy barrier are also required. The reference energy point refers to a certain total energy you are interested in; it is only used to display the rate at this point on the screen and is also a required item. Hindered rotors and rotational constants are optional.
