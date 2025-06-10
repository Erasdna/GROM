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

# Code adapted and reformatted from: https://github.com/larswd/MPET-model-iNPH/blob/main/MPET_brain.py#L403
# This a steady implementation of this model with corresponding simplifications
# We have also adapted the code to work in dolfinx


class SteadyMPET(Problem):
    def __init__(self, mesher):
        super().__init__(mesher)

        self.unit_conversion = {
            "mmHg": 133.32,  # 1 mmHg = 133.32 Pa
            "mmHg/(mL/min)": 133.32 * 1e6 * 60,  # 1 mmHg/(mL/min) = 133*1e6*60 Pa/s
        }

        # ! Check that config and mesh are self-consistent!
        self.index_convention = {
            "arterial": 0,
            "capillary": 1,
            "venous": 2,
            "periarterial": 3,
            "pericapilar": 4,
            "perivenous": 5,
            "extracellular": 6,
        }

        # Load mesh and associate with compartment
        self.make_const = lambda val: fem.Constant(
            self.mesher.domain, PETSc.ScalarType(val)
        )
        self.ds = self.mesher.ds()
        self.dx = self.mesher.dx()
        self._setup_compartments()

        self.betas = np.array([1e-3, 1e-3, 1e-7])
        self.Q = {"prod": 0.33 * 1e3 / 60, "infusion": 1.5 * 1e3 / 60.0}
        self.Q["in"] = self.Q["prod"] + self.Q["infusion"]
        self.C = self.make_const(np.array([1e-4, 1e-8, 1e-4, 1e-8, 1e-8, 1e-8, 1e-8]))
        self.pressures = {
            "AG": self.make_const(8.4 * self.unit_conversion["mmHg"]),
            "CSF": self.make_const(10.0 * self.unit_conversion["mmHg"]),
            "crib": self.make_const(0.0),
        }

        self.R = {"out": 11.0098 - 1.5294}
        self.R["DS"] = 10.81 * self.unit_conversion["mmHg/(mL/min)"] * 1
        self.R["crib"] = 67 * self.unit_conversion["mmHg/(mL/min)"] * 1

        self.spaces = []
        self.maps = []

        for i in range(self.V.num_sub_spaces):
            space_tmp, map_tmp = self.V.sub(i).collapse()
            self.spaces.append(space_tmp)
            self.maps.append(map_tmp)

    def _init_functionspace(self, domain, order=1):
        # Define correct surfaces
        Nc = 7
        el = element("CG", domain.basix_cell(), order)

        V_el = mixed_element([el] * Nc)
        V = fem.functionspace(domain, V_el)
        return V

    def _solve(self, infusion=False, basis=None, return_b_norm=False) -> fem.Function:
        # Compute CSF pressure
        if infusion:
            Q = self.Q["in"]
        else:
            Q = self.Q["prod"]
        coef = 1 / self.R["crib"] + 1 / self.R["DS"]
        self.pressures["CSF"].value = (1 / coef) * (
            Q * 1e-9
            + self.pressures["crib"].value / self.R["crib"]
            + self.pressures["AG"].value / self.R["DS"]
        )

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        uh = fem.Function(self.V)

        # Setup linear system
        F = self._setup_bilinear_form(u, v)
        A = assemble_matrix(fem.form(ufl.lhs(F)))
        A.assemble()
        b = assemble_vector(fem.form(ufl.rhs(F)))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        if basis is not None:
            mat = PETSc.Mat().createDense(basis.shape, array=basis)
            A = A.ptap(mat)
            tmp = np.zeros(basis.shape[-1])
            new_b = PETSc.Vec().createWithArray(tmp, size=tmp.shape[0])
            mat.multTranspose(b, new_b)

        # Initialize solver and settings
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

        # ISF_2_PVSa, PVSA_2_PVSc, PVSA_2_A = self._update_production(u)
        # We ignore the Q_pvs mass conservation constraint in Dreyer 2024 as it seems to be negligible compared to the production/infusion terms
        # (Q - 1000 / 60 * abs(PVSA_2_PVSc + ISF_2_PVSa + PVSA_2_A)) * 1e-9
        if return_b_norm:
            return uh, b.norm()

        return uh

    def _setup_bilinear_form(self, u, v):
        # Write system of equations
        F = 0
        for i in range(self.V.num_sub_spaces):
            F += (
                self.kappa[i]
                * ufl.inner(ufl.grad(u[i]), ufl.grad(v[i]))
                * self.dx
                # + self.C[i] * ufl.inner(u[i] - u_n[i], v[i]) / dt * self.dx
            )
            for j in range(self.V.num_sub_spaces):
                if i != j:
                    F += self.w[i, j] * ufl.inner(u[i] - u[j], v[i]) * self.dx

        # Boundary conditions
        F += (
            (-1) * self.b_avg * v[0] * self.pial_ds
            + self.Q["prod"]
            / self.volumes["areas"]["ventricles"]
            * v[1]
            * self.ventricle_ds
            - self.betas[0] * (self.pressures["CSF"] - u[3]) * v[3] * self.pial_ds
            - self.betas[1]
            * ((self.pressures["AG"] + self.pressures["CSF"]) / 2 - u[2])
            * v[2]
            * self.pial_ds
            - self.betas[2]
            * ((self.pressures["AG"] + self.pressures["CSF"]) / 2 - u[5])
            * v[5]
            * self.pial_ds
        )

        return F

    def bc_correction(self, u_candidate):
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        uh = fem.Function(self.V)

        n = ufl.FacetNormal(self.mesher.domain)
        F = (
            ufl.inner((ufl.grad(u) * n), v) * (self.pial_ds + self.ventricle_ds)
            + ufl.inner(u, v) * self.dx
        )
        F += (-1)*(
            (-1) * self.b_avg * v[0] * self.pial_ds
            + self.Q["prod"]
            / self.volumes["areas"]["ventricles"]
            * v[1]
            * self.ventricle_ds
            - self.betas[0] * (self.pressures["CSF"] - u[3]) * v[3] * self.pial_ds
            - self.betas[1]
            * ((self.pressures["AG"] + self.pressures["CSF"]) / 2 - u[2])
            * v[2]
            * self.pial_ds
            - self.betas[2]
            * ((self.pressures["AG"] + self.pressures["CSF"]) / 2 - u[5])
            * v[5]
            * self.pial_ds
        ) - ufl.inner(u_candidate, v) * self.dx

        A = assemble_matrix(fem.form(ufl.lhs(F)))
        A.assemble()
        b = assemble_vector(fem.form(ufl.rhs(F)))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # Initialize solver and settings
        solver = PETSc.KSP().create(self.mesher.domain.comm)
        solver.setOperators(A)

        solver.setType(PETSc.KSP.Type.GMRES)
        solver.getPC().setType(PETSc.PC.Type.HYPRE)
        solver.solve(b, uh.x.petsc_vec)
        print(
            f"Converged Reason {solver.getConvergedReason()}"
            + f"\nNumber of iterations {solver.getIterationNumber()}"
        )
        uh.x.scatter_forward()
        return uh

    def _setup_compartments(self):
        # We setup and initialize a large number of physical quantities

        # Setup porosity relative to cerebral blod volume fraction
        CBVf = 0.033
        self.porosity = np.array(
            [
                0.33 * CBVf,
                0.1 * CBVf,
                0.57 * CBVf,
                1.4 * 0.33 * CBVf,
                0.1 * CBVf,
                1.4 * 0.57 * CBVf,
                0.14,
            ]
        )

        # Setup relevant fluid parameters
        self.fluid_parameters = dict(
            {
                "rho_f": fem.Constant(self.mesher.domain, PETSc.ScalarType(1e-3)),
                "nu_f": fem.Constant(self.mesher.domain, PETSc.ScalarType(0.75)),
            }
        )
        self.fluid_parameters["mu_f"] = fem.Constant(
            self.mesher.domain,
            PETSc.ScalarType(
                self.fluid_parameters["rho_f"].value
                * self.fluid_parameters["nu_f"].value
            ),
        )
        self.fluid_parameters["mu_b"] = fem.Constant(
            self.mesher.domain,
            PETSc.ScalarType(self.fluid_parameters["mu_f"].value * 3),
        )

        # Compute resistivity
        self.resistivity = np.array(
            [
                0.000939857688751 * self.unit_conversion["mmHg/(mL/min)"],
                1,
                8.14915973766e-05 * self.unit_conversion["mmHg/(mL/min)"],
                1.02 * self.unit_conversion["mmHg/(mL/min)"],
                125 * self.unit_conversion["mmHg/(mL/min)"],
                0.079 * self.unit_conversion["mmHg/(mL/min)"],
                0.57 * self.unit_conversion["mmHg/(mL/min)"],
            ]
        )

        # Inter endfeet gaps
        self.IEG = {
            "venous": 0.64 * self.unit_conversion["mmHg/(mL/min)"],
            "arterial": 0.57 * self.unit_conversion["mmHg/(mL/min)"],
        }

        conductivity_ECS = 20 * 1e-18 / self.fluid_parameters["mu_f"].value
        constant = (
            self.resistivity[-1]
            * conductivity_ECS
            / self.fluid_parameters["mu_f"].value
        )
        # Compute permeability
        self.kappa = np.array(
            [
                1e6
                * constant
                * self.fluid_parameters["mu_b"].value
                / self.resistivity[0],
                1e6 * 1.44e-15 / self.fluid_parameters["mu_b"].value,
                1e6
                * constant
                * self.fluid_parameters["mu_b"].value
                / self.resistivity[2],
                1e6
                * constant
                * self.fluid_parameters["mu_f"].value
                / self.resistivity[3],
                1e6
                * constant
                * self.fluid_parameters["mu_f"].value
                / self.resistivity[4],
                1e6
                * constant
                * self.fluid_parameters["mu_f"].value
                / self.resistivity[5],
                1e6 * conductivity_ECS,
            ]
        )
        self.kappa = self.make_const(self.kappa)

        self._compute_volumes()

        self.blood_flow = 712.5
        self.b_avg = self.make_const(
            self.blood_flow * 1e3 / 60.0 / self.volumes["areas"]["pia"]
        )

        # Compute transfer coefficients
        w_tmp = np.zeros((self.V.num_sub_spaces, self.V.num_sub_spaces))

        w_tmp[self.index_convention["arterial"], self.index_convention["capillary"]] = (
            self.b_avg.value
            * self.volumes["areas"]["pia"]
            / (self.volumes["volume"]["brain"] * 60 * self.unit_conversion["mmHg"])
        )
        w_tmp[self.index_convention["capillary"], self.index_convention["venous"]] = (
            self.b_avg.value
            * self.volumes["areas"]["pia"]
            / (self.volumes["volume"]["brain"] * 12.5 * self.unit_conversion["mmHg"])
        )
        w_tmp[
            self.index_convention["pericapilar"], self.index_convention["capillary"]
        ] = 1 / (
            self.volumes["volume"]["brain"]
            * 1e-9
            * self.resistivity[self.index_convention["pericapilar"]]
        )
        w_tmp[self.index_convention["perivenous"], self.index_convention["venous"]] = (
            1e-15
        )
        w_tmp[
            self.index_convention["periarterial"], self.index_convention["arterial"]
        ] = 1e-15
        w_tmp[
            self.index_convention["periarterial"], self.index_convention["pericapilar"]
        ] = 1e-6
        w_tmp[
            self.index_convention["periarterial"], self.index_convention["perivenous"]
        ] = 1e-6
        w_tmp[
            self.index_convention["perivenous"], self.index_convention["extracellular"]
        ] = 1 / (self.volumes["volume"]["brain"] * 1e-9 * self.IEG["venous"])
        w_tmp[
            self.index_convention["periarterial"],
            self.index_convention["extracellular"],
        ] = 1 / (self.volumes["volume"]["brain"] * 1e-9 * self.IEG["arterial"])
        w_tmp[
            self.index_convention["pericapilar"],
            self.index_convention["extracellular"],
        ] = 1e-10

        self.w = self.make_const(w_tmp + w_tmp.T)

    def _update_production(self, u):
        scaling = 60 / 1000
        local_form = []
        ret = []

        # ISF_2_PVSa, ISF_2_PVSa and PVSA_2_PVSc
        for ind in [[3, 6], [3, 4], [3, 0]]:
            local_form.append(
                scaling
                * fem.assemble_scalar(
                    fem.form(self.w[*ind] * (u[ind[0]] - u[ind[1]]) * self.dx)
                )
            )
        for loc in local_form:
            ret.append(self.mesher.domain.comm.allreduce(loc))

        return ret

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
        for i, name in zip(range(7), self.index_convention.keys()):
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