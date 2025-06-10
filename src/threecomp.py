from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
)
from basix.ufl import element, mixed_element
from dolfinx import fem
import ufl
from petsc4py import PETSc
import numpy as np
from src.Problem import Problem
from petsc4py import PETSc
from src.fenics_utils import norm


class threecomp(Problem):
    def __init__(self, mesher):
        super().__init__(mesher)

        self.index_convention = {
            "extracellular": 0,
            "perivascular": 1,
            # "blood": 2,
        }

        # Load mesh and associate with compartment
        self.make_const = lambda val: fem.Constant(
            self.mesher.domain, PETSc.ScalarType(val)
        )
        self.ds = self.mesher.ds()
        self.dx = self.mesher.dx()

        # extracellular, perivascular and blood compartments
        self.compartments = ["e", "p", "b"]

        # Set up (dimensionless) quantities
        self.n = {
            "e": 0.2,
            "p": 0.02,
            "b": 0.05,
        }
        self.D = {"e": 1.3e-4, "p": 3.9e-4}  # 11.2,  # 33.7,
        self.r = {
            "b": 3.6,
        }
        self.pi = {
            "ep": 2.9e-2,  # 25,
            "pb": 0.2e-7,  # 0.17,
        }
        self.k = {"e": 1.0e-5, "p": 3.7e-4}  # 0.8,  # 32,

        self.BV = 5e6  # Default to human
        self._compute_volumes()

        self.spaces = []
        self.maps = []

        for i in range(self.V.num_sub_spaces):
            space_tmp, map_tmp = self.V.sub(i).collapse()
            self.spaces.append(space_tmp)
            self.maps.append(map_tmp)

    def _init_functionspace(self, domain, order=2):
        # Define correct surfaces
        el = element("CG", domain.basix_cell(), order)
        V_el = mixed_element([el, el])
        V = fem.functionspace(domain, V_el)
        return V

    def _solve(self, basis=None):
        n_e, n_p, _ = self.n.values()
        De, Dp = self.D.values()
        ke, kp = self.k.values()
        pi_ep, pi_pb = self.pi.values()
        
        tau_1 = 4.43e4
        tau_2 = 8.4e4
        a_sas = 0.52
        a_vent = 0.2
        phi = 0.2
        sas_func = lambda t, a: a / phi * (-np.exp(-t / tau_1) + np.exp(-t / tau_2))
        t_equil = np.log(tau_2 / tau_1) / (1 / tau_1 - 1 / tau_2)

        c_sas_pial = fem.Constant(
            self.mesher.domain, PETSc.ScalarType(sas_func(t_equil, a_sas))
        )

        c_sas_vent = fem.Constant(
            self.mesher.domain, PETSc.ScalarType(sas_func(t_equil, a_vent))
        )

        ce, cp = ufl.TrialFunctions(self.V)
        ve, vp = ufl.TestFunctions(self.V)

        a = (
            +ufl.inner((n_e * De * ufl.grad(ce)), ufl.grad(ve)) * self.dx
            + ufl.inner((n_p * Dp * ufl.grad(cp)), ufl.grad(vp)) * self.dx
            + pi_ep * (ce - cp) * (ve - vp) * self.dx
            + pi_pb * cp * vp * self.dx
            + ke * ce * ve * n_e * (self.pial_ds + self.ventricle_ds)
            + kp * cp * vp * n_p * (self.pial_ds + self.ventricle_ds)
        )
        l = (
            (ke * c_sas_pial * ve * n_e) * self.pial_ds
            + (ke * c_sas_vent * ve * n_e) * self.ventricle_ds
            + (kp * c_sas_pial * vp * n_p) * self.pial_ds
            + (kp * c_sas_vent * vp * n_p) * self.ventricle_ds
        )

        A = assemble_matrix(fem.form(a))
        A.assemble()
        b = assemble_vector(fem.form(l))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        if basis is not None:
            mat = PETSc.Mat().createDense(basis.shape, array=basis)
            A = A.ptap(mat)
            tmp = np.zeros(basis.shape[-1])
            new_b = PETSc.Vec().createWithArray(tmp, size=tmp.shape[0])
            mat.multTranspose(b, new_b)

        uh = fem.Function(self.V)
        solver = PETSc.KSP().create(self.mesher.domain.comm)
        solver.setOperators(A)
        # Run solver
        if basis is not None:
            sol_numpy = np.zeros(basis.shape[-1])
            sol = PETSc.Vec().createWithArray(sol_numpy, size=sol_numpy.shape[0])
            solver.setType(PETSc.KSP.Type.LSQR)
            solver.solve(new_b, sol)
            mat.mult(sol, uh.x.petsc_vec)
        else:
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.getPC().setType(PETSc.PC.Type.HYPRE)
            solver.solve(b, uh.x.petsc_vec)
      
        print(
            f"Converged Reason {solver.getConvergedReason()}"
            + f"\nNumber of iterations {solver.getIterationNumber()}"
        )
        uh.x.scatter_forward()

        return uh

    def compute_error(self, target_solution, candidate_solution):
        return_dict = {}
        comm = self.mesher.domain.comm

        def err(reduced, target):
            L2 = norm(reduced - target, "L2", comm) / norm(target, "L2", comm)
            H1 = norm(reduced - target, "H1", comm) / norm(target, "H1", comm)
            return L2, H1

        full_dict = {}
        L2, H1 = err(candidate_solution, target_solution)
        full_dict["L2"] = L2
        full_dict["H1"] = H1
        return_dict["all"] = full_dict

        if self.mesher.domain.comm.Get_rank() == 0:
            print("{:<15} {:>20} {:>20}".format("Compartment", "L2", "H1"))

        compartments = {}
        for i, name in zip(range(self.V.num_sub_spaces), self.index_convention.keys()):
            reduced = candidate_solution.sub(i)
            target = target_solution.sub(i)

            L2, H1 = err(reduced, target)

            comp = {}
            comp["L2"] = L2
            comp["H1"] = H1
            compartments[name] = comp.copy()

            if self.mesher.domain.comm.Get_rank() == 0:
                print(
                    "{:<15} {:>20} {:>20}".format(
                        name,
                        "{:.3E}".format(L2),
                        "{:.3E}".format(H1),
                    )
                )
        return_dict["compartments"] = compartments
        return return_dict
