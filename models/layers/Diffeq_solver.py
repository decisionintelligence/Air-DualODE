import time

class DiffeqSolver:
    def __init__(self, method, odeint_rtol=1e-5,
                 odeint_atol=1e-5, adjoint=False):
        self.ode_method = method
        if adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
        self.odeint = odeint

        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def solve(self, odefunc, first_point, time_steps_to_pred):
        """
        Decoder the trajectory through the ODE Solver.
        :param time_steps_to_pred: horizon
        :param first_point: (batch_size, num_nodes * latent_dim)
        :return: pred_y: # shape (horizon, n_traj_samples, batch_size, self.num_nodes * self.output_dim)
        """
        start_time = time.time()
        odefunc.nfe = 0
        pred_y = self.odeint(odefunc,
                             first_point,
                             time_steps_to_pred,
                             rtol=self.rtol,
                             atol=self.atol,
                             method=self.ode_method)
        # pred_y: (seq_len + 1[first point]) x B x N
        time_fe = time.time() - start_time

        return pred_y, (odefunc.nfe, time_fe)
