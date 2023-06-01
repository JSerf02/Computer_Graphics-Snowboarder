import numpy as np
import bisect
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class BSpline:
    def __init__(self, t, c, d):
        """
        t = knots
        c = bspline coeff
        d = bspline degree
        """
        self.t = t
        self.c = c
        self.d = d
        assert self.is_valid()
    
    def is_valid(self) -> bool:
    #TODO: Q? -- complete this function.
        return len(self.t) == len(self.c) + self.d + 1 and self.d >= 1 and len(self.t) >= 1 and len(self.c) >= 1

    def bases(self, x, k, i):
        # TODO: Q? -- complete this function
        x = x % self.t[len(self.t) - 1]
        if not hasattr(self, "cur_bases") or self.cur_bases[0] != x:
            self.cur_bases = (x, {}) # Stores an internal dictionary of calculated bases to improve speed at the cost of space
        if (i, k) in self.cur_bases[1]:
            return self.cur_bases[1][(i, k)]
        if k == 1:
            b_ik = 1 if self.t[i] <= x < self.t[i + 1] else 0
        else:
            t_i = self.t[i]
            t_ik = self.t[i + k]
            b_ik = (x - t_i) / (self.t[i + k - 1] - t_i) * self.bases(x, k - 1, i) + (t_ik - x) / (t_ik - self.t[i + 1]) * self.bases(x, k - 1, i + 1)
        self.cur_bases[1][(i, k)] = b_ik
        return b_ik

    def interp(self, x):
        # TODO: Q? -- complete this function
        result = 0.0 * self.c[0] # Ensures the proper type of 0 (ex: 0 * (5, 3, 1) = (0, 0, 0))
        for i in range(len(self.c)):
            result += self.c[i] * self.bases(x, self.d + 1, i)
        return result
    
class BezierSpline:
    def __init__(self, ts, coeffs, degree):
        self.ts = ts
        self.coeffs = coeffs
        self.degree = degree
        assert self.is_valid()
    
    def is_valid(self) -> bool:
        
        if len(self.ts) < 2:
            return False
        
        num_coeffs = len(self.coeffs)
        strictly_increasing = all(i < j for i, j in zip(self.ts, self.ts[1:])) # https://www.geeksforgeeks.org/python-check-if-list-is-strictly-increasing/#

        return strictly_increasing and num_coeffs % self.degree == 1 and num_coeffs == len(self.ts)

    def interp(self, t):
        # Determine the t values to interpolate between
        interp_ts = self.ts[::self.degree]
        end_t_idx = bisect.bisect(interp_ts, t)

        # If t is out of bounds, return the endpoints
        if(end_t_idx == 0) :
            return self.coeffs[0]
        elif (end_t_idx == len(interp_ts)):
            return self.coeffs[len(self.coeffs) - 1]
        
        # Have t represent the percentage it is between t0 and t1
        t0 = interp_ts[end_t_idx - 1]
        t1 = interp_ts[end_t_idx]
        t = (t - t0) / (t1 - t0)
        
        # Determine the current coefficients corresponding to these t values
        start_c_idx = (end_t_idx - 1) * self.degree
        cur_coeffs = [c for c in self.coeffs[start_c_idx : start_c_idx + self.degree + 1]]

        # Apply De_Castlejau's Algorithm (https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm)
        for i in range(1, self.degree + 1):
            for j in range (self.degree + 1 - i):
                cur_coeffs[j] = cur_coeffs[j] * (1 - t) + cur_coeffs[j + 1] * t
        return cur_coeffs[0]

class HermiteSpline(BezierSpline):
    # A spline determined by control points and the first derivatives at those points
    def __init__(self, ts, coeffs):
        pairs = [term for term in coeffs if len(term) == 2] # Only use coeffs that are length 2 (containing a point and a slope)
        
        # Convert to a Bezier Spline as follows: 
        # [(t_i, (p_i, d_i)), (t_(i+1), (p_(i+1), d_(i+1)))] |=> 
        #   [
        #       (t_i, p_i), 
        #       (t_i + (t_i + t_(i+1)) / 3, p_i + d_i * (t_i + t_(i+1)) / 3)
        #       (t_(i+1) - (t_i + t_(i+1)) / 3, p_(i+1) - d_(i+1) * (t_i + t_(i+1)) / 3)
        #       (t_(i+1), p_(i+1))
        #   ]
        bezier_ts = []
        bezier_coeffs = []
        for idx in range(len(pairs) - 1):
            start = np.array(pairs[idx][0])
            start_deriv = np.array(pairs[idx][1])
            end = np.array(pairs[idx + 1][0])
            end_deriv = np.array(pairs[idx + 1][1])

            scaled_delta_t = (ts[idx + 1] - ts[idx]) / 3

            bezier_ts.append(ts[idx])
            bezier_coeffs.append(start)
            
            bezier_ts.append(ts[idx] + scaled_delta_t)
            bezier_coeffs.append(start + scaled_delta_t * start_deriv)

            bezier_ts.append(ts[idx + 1] - scaled_delta_t)
            bezier_coeffs.append(end - scaled_delta_t * end_deriv)
        bezier_ts.append(ts[len(ts) - 1])
        bezier_coeffs.append(pairs[len(pairs) - 1][0])
        
        super().__init__(bezier_ts, bezier_coeffs, 3)
    
    def interp(self, t):
        return super().interp(t)

class CardinalSpline(HermiteSpline):
    # A Hermite spline where the derivative at point p_i is (1-c) * (p_(i+1) - p_(i-1)) / (t_(i+1) - t_(i-1))
    # - c is a constant in [0, 1] which scales the derivatives
    def __init__(self, ts, coeffs, c):
        hermite_ts = ts[1:-1] # First and last points are excluded, as there are no previous/next values to use for the derivative calculation
        hermite_coeffs = []
        for i in range(1, len(coeffs) - 1):
            hermite_coeffs.append((coeffs[i], (1 - c) * (coeffs[i + 1] - coeffs[i - 1]) / (ts[i + 1] - ts[i - 1])))
        super().__init__(hermite_ts, hermite_coeffs)
    
    def interp(self, t):
        return super().interp(t)

class CatmullRomSpline(CardinalSpline):
    # A Cardinal Spline with evenly spaced ts and a c value of 0
    def __init__(self, coeffs):
        ts = np.linspace(-1, len(coeffs) - 2, len(coeffs))
        super().__init__(ts, coeffs, 0)
    
    def interp(self, t):
        return super().interp(t)
    



if __name__ == '__main__':
    # t = [] # set some knots. change this.

    num_points = 10

    control_points = []
    for i in range(num_points + 2):
        control_points.append(np.array([np.random.rand(), np.random.rand(), np.random.rand()]))

    spline = CatmullRomSpline(control_points)

    num_ts = 500
    ts = np.linspace(0, len(control_points) - 3, 500)

    results = []
    for t in ts:
        results.append(spline.interp(t))
    
    xs = list(map((lambda res : res[0]), results))
    ys = list(map((lambda res : res[1]), results))
    zs = list(map((lambda res : res[2]), results))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xs, ys, zs, 'blue')
    ax.set_title('Catmull Rom')

    control_points = control_points[1:-1]
    control_xs = list(map((lambda cont: cont[0]), control_points))
    control_ys = list(map((lambda cont: cont[1]), control_points))
    control_zs = list(map((lambda cont: cont[2]), control_points))
    ax.scatter(control_xs, control_ys, control_zs, control_points)
    plt.show()
