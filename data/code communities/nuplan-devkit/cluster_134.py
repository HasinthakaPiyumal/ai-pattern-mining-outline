# Cluster 134

class TestDeepKinematicUnicycleLayer(unittest.TestCase):
    """
    Test Deep Kinematic Unicycle Layer.
    """

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.dev = torch.device('cpu')
        self.dtype = torch.float
        self.layer = DeepDynamicalSystemLayer(dynamics=KinematicUnicycleLayerRearAxle())
        self.layer.to(device=self.dev)
        self.timestep = 0.5
        self.x0 = torch.tensor([0.1, -0.2, 0.0, 0.7, -0.4, 0.75, 0.0], device=self.dev, dtype=self.dtype)

    def assert_gradient_element_almost_equal(self, x0: torch.Tensor, control: torch.Tensor, manual_grad: torch.Tensor, el: int, control_idx: int) -> None:
        """
        Auxiliary function to ensure single element
        of the gradient is computed correctly.
        """
        ctrl = control.clone().detach().requires_grad_(True)
        xnext = self.layer.forward(x0, ctrl, self.timestep, None)
        xnext[0, el].backward()
        self.assertAlmostEqual(ctrl.grad.detach().cpu()[control_idx, 0].item(), manual_grad[el, 0].item())
        self.assertAlmostEqual(ctrl.grad.detach().cpu()[control_idx, 1].item(), manual_grad[el, 1].item())

    def test_autograd_computation_one_step(self) -> None:
        """
        Test autograd calculations for DeepKinematicLayer (one step prediction)
        """
        curvature = 0.125
        jerk = 0.3
        manual_grad = kinematic_unicycle_rear_axle_manual_grad(curvature, jerk, self.timestep, self.x0)
        control = torch.tensor([curvature, jerk], device=self.dev, dtype=self.dtype).reshape(1, -1).requires_grad_(True)
        for i in range(self.layer.dynamics.state_dim()):
            self.assert_gradient_element_almost_equal(self.x0, control, manual_grad, i, 0)

    def test_autograd_computation_no_future_leak(self) -> None:
        """
        Check states' gradient does not depend
        on future inputs
        """
        curvature = 0.125
        jerk = 0.3
        control = torch.tensor([[curvature, jerk], [curvature, -jerk]], device=self.dev, dtype=self.dtype).requires_grad_(True)
        for i in range(self.layer.dynamics.state_dim()):
            for k in range(control.shape[-2]):
                for j in range(k):
                    ctrl = control.clone().detach().requires_grad_(True)
                    xnext = self.layer.forward(self.x0, ctrl, self.timestep, None)
                    xnext[j, i].backward()
                    self.assertAlmostEqual(ctrl.grad.detach().cpu()[k, 0].item(), 0.0)
                    self.assertAlmostEqual(ctrl.grad.detach().cpu()[k, 1].item(), 0.0)

def kinematic_unicycle_rear_axle_manual_grad(curvature: float, jerk: float, t: float, x0: torch.Tensor) -> torch.Tensor:
    """
    Helper function to manually compute gradient.
    """
    man_grad = torch.zeros(7, 2)
    v0 = torch.sqrt(x0[3] ** 2 + x0[4] ** 2)
    man_grad[2, 0] = t * v0
    man_grad[5, 1] = t * torch.cos(x0[2])
    man_grad[6, 1] = t * torch.sin(x0[2])
    return man_grad

