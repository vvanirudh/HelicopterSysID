# System Identification for Nonlinear Helicopter Dynamics

In this project, we parameterized helicopter dynamics using the following 20-D parameterization:

```python
self.m = params[0]
self.Ixx = params[1]
self.Iyy = params[2]
self.Izz = params[3]
self.Ixy = self.Iyz = self.Ixz = 0.0
self.Tx = params[4:7].copy()
self.Ty = params[7:10].copy()
self.Tz = params[10:13].copy()
self.Fx = params[13]
self.Fy = params[14:16].copy()
self.Fz = params[16:19].copy()
self.g = params[19]
```

This results in a highly nonlinear nonconvex dynamics function in terms of these parameters (look into `helicopter.py` for more details.)

The objective is to learn a policy that makes the helicopter *hover* at origin in the real world where the helicopter dynamics are dictated using the following parameters:

```python
## Mass and inertia
self.m = 5  # kg
self.Ixx = 0.3
self.Iyy = 0.3
self.Izz = 0.3
self.Ixy = self.Ixz = self.Iyz = 0
self.g = 9.81

# Aerodynamic forces parameters
self.Tx = np.array([0, -3.47, 13.20]) * self.Ixx
self.Ty = np.array([0, -3.06, -9.21]) * self.Iyy
self.Tz = np.array([0, -2.58, 14.84]) * self.Izz
self.Fx = -0.048 * self.m
self.Fy = np.array([0, -0.12]) * self.m
self.Fz = np.array([-9.81, -0.0005, -27.5]) * self.m
```

The objective is encoded using a quadratic function as follows:
$$ cost(x, u) = (sum_{t=1}^{H-1} x_t^T Q x_t + u_t^T R u_t) + x_H^T Q_f x_H $$
i.e. we penalize any deviation from origin, and any control effort expended.