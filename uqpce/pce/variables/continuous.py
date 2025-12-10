from builtins import setattr, getattr
import sys
from fractions import Fraction
import math
from multiprocessing import Process, Manager

from sympy import *
import numpy as np
from scipy.stats import norm, beta as fbeta, gamma, expon, uniform, lognorm
from scipy.linalg import pascal
from scipy.integrate import quad

from sympy import symbols, zeros, integrate, N, sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify

from uqpce.pce.enums import Distribution, UncertaintyType
from uqpce.pce._helpers import uniform_hypercube
from uqpce.pce.variables.variable import Variable
from uqpce.pce.error import VariableInputError

try:
    from mpi4py.MPI import (
        COMM_WORLD as MPI_COMM_WORLD, DOUBLE as MPI_DOUBLE, MAX as MPI_MAX
    )
    comm = MPI_COMM_WORLD
    rank = comm.rank
    size = comm.size
    is_manager = (rank == 0)
except:
    comm = None
    rank = 0
    size = 1
    is_manager = True

class ContinuousVariable(Variable):
    """
    Class represents a continuous variable.

    Parameters
    ----------
    pdf :
        the equation that defines the pdf of the variable values
    interval_low :
        the low interval of the variable
    interval_high :
        the high interval of the variable
    order :
        the order of the model to calculate the orthogonal
        polynomials and norm squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """

    def __init__(self, pdf, interval_low, interval_high, order=2, name='', number=0):

        self.distribution = pdf
        self.interval_low = interval_low
        self.interval_high = interval_high
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.low_approx = self.interval_low
        self.high_approx = self.interval_high

        self.type = UncertaintyType.ALEATORY
        # split at white space and rejoin to remove all whitespace- make safer
        self.distribution = ''.join(self.distribution.split())
        self.distribution = (
            parse_expr(self.distribution, local_dict={'x':self.x})
        )

        self.check_num_string()

        self.bounds = (self.low_approx, self.high_approx)
        self.std_bounds = (self.low_approx, self.high_approx)

        self.get_probability_density_func()  # make sure sum over interval = 1

        self.recursive_var_basis(
            self.distribution, self.interval_low, self.interval_high, self.order
        )

        self.create_norm_sq(
            self.interval_low, self.interval_high, self.distribution
        )

        if self.type == UncertaintyType.EPISTEMIC:
            print(
                'The ContinuousVariable is usually not epistemic. For an epistemic '
                'variable, consider using the uniform distribution with type '
                'epistemic.', file=sys.stderr
            )


    def get_probability_density_func(self):
        """
        Turns the input function into the corresponding probability density
        function.
        """
        diff_tol = 1e-5
        tol = 1e-12

        f = lambdify(self.x, self.distribution, ('numpy', 'sympy'))
        const = quad(f, self.low_approx, self.high_approx, epsabs=tol, epsrel=tol)[0]

        const_rnd = np.round(const)

        if np.abs(const_rnd - const) < diff_tol:
            const = const_rnd

        self.distribution = self.distribution / const


    def standardize(self, orig, std_vals):
        """
        For each variable, it adds a new attribute for the standardized values
        from the original input values.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        setattr(self, std_vals, getattr(self, orig))

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        return values  # general variable must already be standardized


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        return value  # general variable must already be standardized


    def check_distribution(self, X):
        """
        Checks all values in an array to ensure that they are standardized.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """

        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and mx > self.high_approx or mn < self.low_approx:
            print(
                f'Large standardized value for variable {self.name} '
                'with user distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1


    def generate_samples(self, samp_size, **kwargs):
        """
        Generates points according to the Latin hypercube; each point is in an
        interval of equal probability.

        Parameters
        ----------
        samp_size :
            the number of points needed to be generated
        """
        decimals = 30

        base = samp_size // size
        rem = samp_size % size
        count = base + (rank < rem)

        ranks = np.arange(0, size, dtype=int)
        seq_count = (ranks < rem) + base
        seq_disp = base * ranks + (ranks >= rem) * rem + (ranks < rem) * ranks

        try:
            y = symbols('y')

            if hasattr(self, 'failed'):
                raise AttributeError  # skip if has already gone through and failed

            # solve for the cumulative density function with 10s timeout
            if is_manager and not hasattr(self, 'cum_dens_func'):
                manager = Manager()
                proc_dict = manager.dict()

                cdf_proc = Process(target=self._calc_cdf, args=(proc_dict,))
                cdf_proc.start()

                cdf_proc.join(10.0)
                if cdf_proc.is_alive():
                    cdf_proc.terminate()

                try:
                    self.cum_dens_func = proc_dict['cum_dens_func']

                except KeyError:
                    self.failed = 1
                    raise ValueError

            # solve for the inverse function with 10s timeout
                inv_proc = Process(target=self._invert, args=(proc_dict,))
                inv_proc.start()

                inv_proc.join(10.0)
                if inv_proc.is_alive():
                    inv_proc.terminate()

                try:
                    self.inverse_func = proc_dict['inverse_func']

                except KeyError:
                    self.failed = 1
                    raise ValueError

            try:
                self.inverse_func = comm.bcast(self.inverse_func, root=0)
            except:
                raise ValueError

            # plug in random uniform 0 -> 1 to solve for x vals
            all_samples = np.zeros(samp_size)

            for i in range(len(self.inverse_func)):  # multiple solutions
                inv_func = (
                    np.vectorize(
                        lambdify(y, str(self.inverse_func[i]), ('numpy', 'sympy'))
                    )
                )

                samples = N(inv_func(uniform_hypercube(0, 1, count)), decimals)

                if comm:
                    comm.Allgatherv(
                        [samples, count, MPI_DOUBLE],
                        [all_samples, seq_count, seq_disp, MPI_DOUBLE]
                    )
                else:
                    all_samples = samples

                if np.min(all_samples) >= self.low_approx and np.max(all_samples) <= self.high_approx:
                    np.random.shuffle(all_samples)
                    return all_samples

            if not (
                (np.min(samples) >= self.low_approx) and (np.max(samples) <= self.high_approx)
            ):
                raise ValueError

        # if cdf or inverse func can't be found, use rejection-acceptance sampling
        except (ValueError, NameError, AttributeError):
            func = lambdify(self.x, self.distribution, ('numpy', 'sympy'))

            try_total = 5000
            tries = try_total // size + (rank < try_total % size)
            max_all = np.zeros(1)

            try:
                max_val = (
                    np.max(func(
                        np.random.uniform(
                            self.low_approx, self.high_approx, tries
                        )
                    ))
                )

            except RuntimeError:
                max_val = np.max(
                    func(
                        np.random.uniform(
                            self.low_approx, self.high_approx, tries
                        )
                    )
                ).astype('float64')

            if comm:
                comm.Allreduce(
                    [max_val, MPI_DOUBLE], [max_all, MPI_DOUBLE], op=MPI_MAX
                )
            else:
                max_all = max_val

            samples = np.zeros(count)
            all_samples = np.zeros(samp_size)

            i = 0
            j = 0

            y_vals = np.random.uniform(0, max_all, count)
            x_vals = np.random.uniform(self.low_approx, self.high_approx, count)
            func_vals = func(x_vals)

            # while loop until all 'samp_size' samples have been generated
            while i < count:

                if j == count:
                    y_vals = np.random.uniform(0, max_all, count)
                    x_vals = np.random.uniform(self.low_approx, self.high_approx, count)
                    func_vals = func(x_vals)
                    j = 0

                if y_vals[j] <= func_vals[j]:
                    samples[i] = x_vals[j]
                    i += 1

                j += 1

            if comm:
                comm.Allgatherv(
                    [samples, count, MPI_DOUBLE],
                    [all_samples, seq_count, seq_disp, MPI_DOUBLE]
                )
            else:
                all_samples = samples

            np.random.shuffle(all_samples)

            return all_samples


    def create_norm_sq(self, low, high, func):
        """
        Calculates the norm squared values up to the order of polynomial
        expansion based on the probability density function and its
        corresponding orthogonal polynomials.

        Parameters
        ----------
        low :
            the low interval bound for the distribution
        high :
            the high interval bound for the distribution
        func :
            the function corresponding to the distribution
        """
        orthopoly_count = len(self.var_orthopoly_vect)
        self.norm_sq_vals = np.zeros(orthopoly_count)

        tries = 2
        zero = 0

        # is rounded off at 50 decimals, requiring two decimals places
        norm_sq_thresh = 1e-49

        for i in range(orthopoly_count):

            proc_dict = {}

            for j in range(tries):

                self._norm_sq(low, high, func, i, j, proc_dict)

                try:
                    if (proc_dict['out'] is not None) and (not math.isclose(proc_dict['out'], zero)):
                        self.norm_sq_vals[i] = proc_dict['out']
                        break  # only breaks inner loop

                except KeyError:
                    pass

        if (self.norm_sq_vals == zero).any():
            print(f'Finding the norm squared for variable {self.name} failed.', file=sys.stderr)

        if (self.norm_sq_vals <= norm_sq_thresh).any():
            print(
                f'At least one norm squared value for variable {self.name} is '
                f'very small. This can introduce error into the model.', file=sys.stderr
            )


    def _norm_sq(self, low, high, func, i, region, proc_dict):
        """
        An assistant to create_norm_sq; allows the norm squared calculations to
        have a timeout if an error isn't raised and the solution isn't found
        reasonably quickly.

        Parameters
        ----------
        low :
            the low interval bound for the distribution
        high :
            the high interval bound for the distribution
        func :
            the function corresponding to the distribution
        i :
            the index of the norm squared to calculate
        region :
            which sympy calculation to try for the norm squared
        proc_dict :
            the dictionary in which the output will be stored
        """
        proc_dict['out'] = None

        # round 0.99999999 to 1 to reduce error; if value is small, don't round
        thresh = 1e-2
        tol = 1e-12
        diff_tol = 1e-8
        decimals = 30

        if high == 'oo':
            ul = np.inf
        elif high == 'pi':
            ul = np.pi
        elif high == '-pi':
            ul = -np.pi
        else:
            ul = high

        if low == '-oo':
            ll = -np.inf
        elif low == 'pi':
            ll = np.pi
        elif low == '-pi':
            ll = -np.pi
        else:
            ll = low

        if region == 0:

            try:
                f = lambdify(self.x, func * self.var_orthopoly_vect[i] ** 2, ('numpy', 'sympy'))
                ans = quad(f, ll, ul, epsabs=tol, epsrel=tol)[0]

                if np.abs(int(ans) - ans) < diff_tol:
                    proc_dict['out'] = int(ans)
                elif ans > thresh:
                    proc_dict['out'] = round(ans, 7)
                else:
                    proc_dict['out'] = ans
            except:
                pass

        elif region == 1:
            try:
                f = lambdify(
                    self.x,
                    N(func * self.var_orthopoly_vect[i] ** 2, decimals),
                    ('numpy', 'sympy')
                )

                ans = quad(f, ll, ul, epsabs=tol, epsrel=tol)[0]

                if np.abs(int(ans) - ans) < diff_tol:
                    proc_dict['out'] = int(ans)
                elif ans > thresh:
                    proc_dict['out'] = round(ans, 7)
                else:
                    proc_dict['out'] = ans
            except:
                pass

        elif region == 2:
            try:
                f = lambdify(
                    self.x,
                    sympify(f'{func} * ({self.var_orthopoly_vect[i]}) ** 2'),
                    ('numpy', 'sympy')
                )

                ans = quad(f, ll, ul, epsabs=tol, epsrel=tol)[0]

                if np.abs(int(ans) - ans) < diff_tol:
                    proc_dict['out'] = int(ans)
                elif ans > thresh:
                    proc_dict['out'] = round(ans, 7)
                else:
                    proc_dict['out'] = ans
            except:
                pass


    def recursive_var_basis(self, func, low, high, order):
        """
        Recursively calculates the variable basis up to the input 'order'.

        Parameters
        ----------
        func :
            the probability density function of the input equation
        low :
            the low bound on the variable
        high :
            the high bound on the variable
        order :
            the order of polynomial expansion
        """
        tol = 1e-12

        if low == '-oo':
            low = -np.inf
        if high == 'oo':
            high = np.inf

        if order == 0:
            self.poly_denom = np.zeros(self.order, dtype=object)
            self.var_orthopoly_vect = np.zeros(self.order + 1, dtype=object)
            self.var_orthopoly_vect[order] = 1
            return

        else:
            self.recursive_var_basis(func, low, high, order - 1)
            curr = self.x ** order

            for i in range(order):
                orthopoly = self.var_orthopoly_vect[i]

                if self.poly_denom[i] == 0:
                    f = lambdify(self.x, orthopoly ** 2 * func, ('numpy', 'sympy'))
                    self.poly_denom[i] = quad(f, low, high, epsabs=tol, epsrel=tol)[0]

                f = lambdify(self.x, self.x ** order * orthopoly * func, ('numpy', 'sympy'))
                intergal_eval = (
                    quad(f, low, high, epsabs=tol, epsrel=tol)[0]
                    / self.poly_denom[i]
                ) * orthopoly

                curr -= intergal_eval

            self.var_orthopoly_vect[order] = curr

            if order == self.order and (self.var_orthopoly_vect == 0).any():
                print(
                    f'Variable {self.name} has at least one orthogonal polynomial '
                    f'that is zero. The model may not be accurate.', file=sys.stderr
                )

            return


    def get_resamp_vals(self, samp_size):
        """
        Generates samp_num number of samples according to the pdf of the
        Variable.

        Parameters
        ----------
        samp_size :
            the number of samples to generate according to the distribution
        """
        return self.generate_samples(samp_size)


    def _calc_cdf(self, proc_dict):
        """
        Calculates the cumulative density function of the distribution.

        Parameters
        ----------
        proc_dict :
            the dictionary in which the output will be stored
        """
        try:
            proc_dict['cum_dens_func'] = integrate(
                self.distribution, (self.x, self.interval_low, self.x)
            )

        except RuntimeError:
            pass

    def _invert(self, proc_dict):
        """
        Solves for the inverse function of the cumulative density function.

        Parameters
        ----------
        proc_dict :
            the dictionary in which the output will be stored
    
        """
        y = symbols('y')

        try:
            proc_dict['inverse_func'] = solve(f'{self.cum_dens_func}-y', self.x)

        except (NameError, NotImplementedError, AttributeError, RuntimeError):
            pass


    def check_num_string(self):
        """
        Checks for values in the input file that correspond to pi, -oo, or oo.
        If these values exist, they are converted into values that Python can
        use to create resampling points.
        """
        decimals = 30

        if self.interval_low == '-oo' or self.interval_high == 'oo':
            x = self.x
            integrate_tuple = (x, self.interval_low, self.interval_high)

            self.mean = integrate(x * self.distribution, integrate_tuple)

            stdev = (
                math.sqrt(
                    integrate(x ** 2 * self.distribution, integrate_tuple)
                    -self.mean ** 2
                )
            )

        if isinstance(self.interval_low, str):

            if 'pi' in self.interval_low:
                temp_low = float(self.interval_low.replace('pi', str(np.pi)))
                self.interval_low = temp_low
                self.low_approx = temp_low

            elif self.interval_low == '-oo':
                self.low_approx = float(N(self.mean - 10 * stdev, decimals))

        if isinstance(self.interval_high, str):

            if 'pi' in self.interval_high:
                temp_high = float(self.interval_high.replace('pi', str(np.pi)))
                self.interval_high = temp_high
                self.high_approx = temp_high

            elif self.interval_high == 'oo':
                self.high_approx = float(N(self.mean + 10 * stdev, decimals))


    def get_mean(self):
        """
        Return the mean of the variable.
        """
        decimals = 30
        if not hasattr(self, 'mean'):
            x = self.x
            integrate_tuple = (x, self.interval_low, self.interval_high)
            self.mean = N(
                integrate(x * self.distribution, integrate_tuple), decimals
            )

        return self.mean

class UniformVariable(ContinuousVariable):
    """
    Represents a uniform variable. The methods in this class correspond to
    those of a uniform variable.

    Parameters
    ----------
    interval_low :
        the low interval of the variable
    interval_high :
        the high interval of the variable
    order :
        the order of the model to calculate the orthogonal polynomials and norm 
        squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """

    def __init__(
            self, interval_low, interval_high, order=2, name='', number=0
        ):

        if not interval_low < interval_high:
            raise VariableInputError(
                'UniformVariable interval_high must be greater than interval_low.'
            )

        self.interval_low = interval_low
        self.interval_high = interval_high
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.UNIFORM
        self.type = UncertaintyType.ALEATORY
        self.dist = uniform(
            loc=self.interval_low, scale=self.interval_high-self.interval_low
        )

        self.generate_orthopoly()

        self.low_approx = self.interval_low
        self.high_approx = self.interval_high

        self.bounds = (self.interval_low, self.interval_high)
        self.std_bounds = (-1, 1)

        self.check_num_string()

    def generate_samples(self, count, standardize=False):
        """
        Overrides the Variable class generate_samples to align with
        a uniform distribution.

        Parameters
        ----------
        count :
            the number of points needed to be generated
        standardize :
            boolean for generating samples for the standardized version of the
            variable
        """
        return super(ContinuousVariable, self).generate_samples(
            count, standardize=standardize
        )

    def generate_orthopoly(self):
        """
        Generates the orthogonal polynomials for a uniform variable up to the
        order of polynomial expansion.
        """
        self.var_orthopoly_vect = np.zeros(self.order + 1, dtype=object)
        x = self.x
        for n in range(self.order + 1):

            if n == 0:
                self.var_orthopoly_vect[n] = 1

            elif n == 1:
                self.var_orthopoly_vect[n] = x

            else:
                self.var_orthopoly_vect[n] = (
                    (
                        (2 * n - 1) * x
                        * self.var_orthopoly_vect[n - 1] - (n - 1)
                        * self.var_orthopoly_vect[n - 2]
                    )
                / n
            )

    def resample(self, count):
        samps = self.generate_samples(count, standardize=True)
        samps[np.argmin(samps)] = -1
        samps[np.argmax(samps)] = 1
        return samps

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with
        a uniform distribution.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        original = getattr(self, orig)

        standard = self.standardize_points(original)

        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        mean = (
            (self.interval_high - self.interval_low) / 2 + self.interval_low
        )

        stdev = (self.interval_high - self.interval_low) / 2

        return (values - mean) / stdev


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        shift = (
            (self.interval_high - self.interval_low) / 2 + self.interval_low
        )

        factor = (self.interval_high - self.interval_low) / 2

        return (value * factor) + shift


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with
        a uniform distribution.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if (mx > 1 + 1e-5) or (mn < -1 - 1e-5):
            print(
                f'Standardized value for variable {self.name} with uniform '
                'distribution outside expected [-1, 1] bounds.', file=sys.stderr
            )
            return -1


    def get_norm_sq_val(self, matrix_val):
        """
        Overrides the Variable class get_norm_sq_val to align with
        a uniform distribution.

        Parameters
        ----------
        matrix_val :
            the value in the model matrix to consider
        """
        return 1.0 / (2.0 * matrix_val + 1.0)


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the
        values that might contain it.
        """
        if isinstance(self.interval_low, str) and 'pi' in self.interval_low:
            self.interval_low = float(self.interval_low.replace('pi', str(np.pi)))

        if isinstance(self.interval_high, str) and 'pi' in self.interval_high:
            self.interval_high = float(self.interval_high.replace('pi', str(np.pi)))


    def get_mean(self):
        return (self.interval_high - self.interval_low) / 2 + self.interval_low

class NormalVariable(ContinuousVariable):
    """
    Represents a normal variable. The methods in this class correspond to
    those of a normal variable.

    Parameters
    ----------
    mean :
        the mean of the variable
    stdev :
        the standard deviation of the variable
    order :
        the order of the model to calculate the orthogonal polynomials and norm 
        squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """

    __slots__ = ('mean', 'stdev')

    def __init__(self, mean, stdev, order=2, name='', number=0):

        if not stdev > 0:
            raise VariableInputError(
                'NormalVariable stdev must be greater than 0.'
            )

        self.mean = mean
        self.stdev = stdev
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.NORMAL
        self.type = UncertaintyType.ALEATORY

        self.generate_orthopoly()

        low_percent = 8e-17
        high_percent = 1 - low_percent

        self.dist = norm(loc=self.mean, scale=self.stdev)
        self.low_approx = self.dist.ppf(low_percent)
        self.high_approx = self.dist.ppf(high_percent)

        self.std_bounds = (
            self.standardize_points(self.low_approx),
            self.standardize_points(self.high_approx)
        )

        self.bounds = (self.dist.ppf(low_percent), self.dist.ppf(high_percent))

        self.check_num_string()


    def generate_orthopoly(self):
        """
        Generates the orthogonal polynomials for a normal variable up to the
        order of polynomial expansion.
        """
        self.var_orthopoly_vect = zeros(self.order + 1, 1)
        x = self.x

        for n in range(self.order + 1):
            if n == 0:
                self.var_orthopoly_vect[n] = 1

            elif n == 1:
                self.var_orthopoly_vect[n] = 2 * x

            else:
                self.var_orthopoly_vect[n] = (
                    2 * x * self.var_orthopoly_vect[n - 1] - 2 * (n - 1)
                    * self.var_orthopoly_vect[n - 2]
                )

        for n in range(self.order + 1):  # transform into probabalists Hermite poly
            self.var_orthopoly_vect[n] = (
                2 ** (-n / 2)
                * self.var_orthopoly_vect[n].subs({x:x / math.sqrt(2)})
            )

        self.var_orthopoly_vect = np.array(self.var_orthopoly_vect).astype(object).T[0]


    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with
        a normal distribution.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        original = getattr(self, orig)
        standard = self.standardize_points(original)
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        return (values - self.mean) / self.stdev


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        return (value * self.stdev) + self.mean


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with
        a normal distribution.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """

        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mx > 4.5) or (mn < -4.5):
            print(
                f'Large standardized value for variable {self.name} '
                'with normal distribution found. Check input and run matrix.',
                file=sys.stderr
            )
            return -1


    def get_norm_sq_val(self, matrix_value):
        """
        Overrides the Variable class get_norm_sq_val to align with
        a normal distribution.

        Parameters
        ----------
        matrix_val :
            the value in the model matrix to consider
        """
        return math.factorial(matrix_value)


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the
        values that might contain it.
        """

        if isinstance(self.mean, str) and 'pi' in self.mean:
            self.mean = float(self.mean.replace('pi', str(np.pi)))

        if isinstance(self.stdev, str) and  'pi' in self.stdev:
            self.stdev = float(self.stdev.replace('pi', str(np.pi)))

    def generate_samples(self, count, standardize=False):
        """
        Overrides the Variable class generate_samples to align with
        a normal distribution.

        Parameters
        ----------
        count :
            the number of points needed to be generated
        standardize :
            boolean for generating samples for the standardized version of the
            variable
        """
        return super(ContinuousVariable, self).generate_samples(
            count, standardize=standardize
        )
    
    def get_mean(self):
        return self.mean

class BetaVariable(ContinuousVariable):
    """
    Represents a beta variable. The methods in this class correspond to
    those of a beta variable.

    Parameters
    ----------
    alpha :
        the alpha parameter of the variable
    beta :
        the beta parameter of the variable
    interval_low :
        the low interval of the variable
    interval_high :
        the high interval of the variable
    order :
        the order of the model to calculate the orthogonal polynomials and norm 
        squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """
    __slots__ = ('alpha', 'beta')

    equation = '((A+B-1)! * (x)**(A-1) * (1-x)**(B-1)) / ((A-1)! * (B-1)!)'

    def __init__(
            self, alpha, beta, interval_low=0.0, interval_high=1.0, order=2,
            name='', number=0
        ):

        if not (
            (interval_low is self.__init__.__defaults__[0])
            == (interval_high is self.__init__.__defaults__[1])
        ):
            raise VariableInputError(
                'For BetaVariable, if interval_low or interval_high is '
                'provided, both must be provided.'
            )

        if not ((alpha > 0) and (beta > 0)):
            raise VariableInputError(
                'BetaVariable alpha and beta must be greater than 0.'
            )

        self.alpha = alpha
        self.beta = beta
        self.interval_low = interval_low
        self.interval_high = interval_high
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.BETA
        self.type = UncertaintyType.ALEATORY

        scale = self.interval_high - self.interval_low
        self.dist = fbeta(
            self.alpha, self.beta, loc=self.interval_low, scale=scale
        )

        low = 0
        high = 1

        self.std_bounds = (self.interval_low, self.interval_high)

        parsed_dist = parse_expr(
            self.equation,
            local_dict={
                'A':parse_expr(str(Fraction(self.alpha))),
                'B':parse_expr(str(Fraction(self.beta))),
                'x':self.x
            }
        )

        self.generate_orthopoly()
        self.create_norm_sq(low, high, parsed_dist)

        self.low_approx = self.interval_low
        self.high_approx = self.interval_high

        self.bounds = (self.interval_low, self.interval_high)

        self.check_num_string()

    def generate_samples(self, count, standardize=False):
        """
        Overrides the Variable class generate_samples to align with
        a beta distribution.

        Parameters
        ----------
        count :
            the number of points needed to be generated
        standardize :
            boolean for generating samples for the standardized version of the
            variable
        """
        return super(ContinuousVariable, self).generate_samples(
            count, standardize=standardize
        )

    def generate_orthopoly(self):
        """
        Generates the orthogonal polynomials for a beta variable up to the
        self.self.order of polynomial expansion.
        """
        var_orthopoly_vect = np.zeros(self.order + 1, dtype=object)
        self.var_orthopoly_vect = np.zeros(self.order + 1, dtype=object)
        x = self.x
        a = parse_expr(str(Fraction(self.alpha)))
        b = parse_expr(str(Fraction(self.beta)))

        decimals = 30

        for n in range(self.order + 1):

            if n == 0:
                var_orthopoly_vect[n] = 1
                self.var_orthopoly_vect[n] = 1

            elif n == 1:
                var_orthopoly_vect[n] = x - (a / (a + b))
                self.var_orthopoly_vect[n] = x - (a / (a + b))

            else:
                var_orthopoly_vect[n] = x ** n
                pasc = pascal(self.order + 1, kind='lower')

                for m in range(n):
                    var_orthopoly_vect[n] -= parse_expr(
                        f'{pasc[n, m]} * ((a+n-1)!*(a+b+2*m-1)!)/((a+m-1)!*(a+b+n+m-1)!)*({var_orthopoly_vect[m]})',
                        local_dict={'a':a, 'b':b, 'n':n, 'm':m, 'x':x}
                    )

                self.var_orthopoly_vect[n] = N(var_orthopoly_vect[n], decimals)

        return self.var_orthopoly_vect


    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with
        a beta distribution.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        original = getattr(self, orig)
        standard = (
            (original[:] - self.interval_low)
            / (self.interval_high - self.interval_low)
            )
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        standard = (
            (values - self.interval_low)
            / (self.interval_high - self.interval_low)
            )

        return standard


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        unscaled_value = value = (
            value * (self.interval_high - self.interval_low)
            +self.interval_low
        )

        return unscaled_value


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with
        an beta distribution.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """
        shift = 8

        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mx > shift) or (mn < -shift):
            print(
                f'Large standardized value for variable {self.name} '
                'with Beta distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the
        values that might contain it.
        """
        if isinstance(self.alpha, str) and 'pi' in self.alpha:
            self.alpha = float(self.alpha.replace('pi', str(np.pi)))

        if isinstance(self.beta, str) and 'pi' in self.beta:
            self.beta = float(self.beta.replace('pi', str(np.pi)))

        if isinstance(self.interval_low, str) and 'pi' in self.interval_low:
            self.interval_low = float(self.interval_low.replace('pi', str(np.pi)))

        if isinstance(self.interval_high, str) and 'pi' in self.interval_high:
            self.interval_high = float(self.interval_high.replace('pi', str(np.pi)))

    def get_mean(self):
        """
        Return the mean of the variable.
        """
        scale = self.interval_high - self.interval_low
        mean = (
            self.interval_low + scale * (self.alpha / (self.alpha + self.beta))
        )

        return mean

class ExponentialVariable(ContinuousVariable):
    """
    Represents an exponential variable. The methods in this class correspond to
    those of an exponential variable.

    Parameters
    ----------
    lambd :
        the lambda parameter of the variable values
    interval_low :
        the low interval of the variable
    order :
        the order of the model to calculate the orthogonal polynomials and norm 
        squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """

    __slots__ = ('lambda')

    equation = 'lambd * exp(-lambd * x)'

    def __init__(self, lambd, interval_low=0, order=2, name='', number=0):

        if lambd <= 0:
            raise VariableInputError(
                'ExponentialVariable lambd must be greater than 0.'
            )

        setattr(self, 'lambda', lambd)
        self.interval_low = interval_low
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.EXPONENTIAL
        self.type = UncertaintyType.ALEATORY
        self.dist = expon(scale=1 / getattr(self, 'lambda'), loc=self.interval_low)

        low = 0
        high = 'oo'

        parsed_dist = parse_expr(
            self.equation,
            local_dict={
                'lambd':parse_expr(str(Fraction(getattr(self, 'lambda')))),
                'x':self.x
            }
        )

        # if inf bounds, find approximate bound
        low_percent = 8e-17
        high_percent = 1 - low_percent
        self.low_approx = self.interval_low
        self.high_approx = self.dist.ppf(high_percent)

        self.bounds = (self.interval_low, self.high_approx)
        self.std_bounds = (low, self.standardize_points(self.high_approx))

        self.recursive_var_basis(parsed_dist, low, high, self.order)
        self.create_norm_sq(low, high, parsed_dist)

        self.check_num_string()
    
    def generate_samples(self, count, standardize=False):
        """
        Overrides the Variable class generate_samples to align with
        a exponential distribution.

        Parameters
        ----------
        count :
            the number of points needed to be generated
        standardize :
            boolean for generating samples for the standardized version of the
            variable
        """
        return super(ContinuousVariable, self).generate_samples(
            count, standardize=standardize
        )

    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with an exponential distribution.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        original = getattr(self, orig)
        standard = (original[:] - self.interval_low)
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        return values - self.interval_low


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        return value + self.interval_low


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with
        an exponential distribution.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """
        shift = 15

        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mn < 0) or (mx > shift):
            print(
                f'Large standardized value for variable {self.name} '
                'with exponential distribution found. Check input and run '
                'matrix.', file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the
        values that might contain it.
        """
        lambd = getattr(self, 'lambda')

        if isinstance(lambd, str) and 'pi' in lambd:
            setattr(self, 'lambda', float(lambd.replace('pi', str(np.pi))))

    def get_mean(self):
        """
        Return the mean of the variable.
        """
        return self.interval_low + (1 / getattr(self, 'lambda'))

class GammaVariable(ContinuousVariable):
    """
    Represents a gamma variable. The methods in this class correspond to
    those of a gamma variable.

    Parameters
    ----------
    alpha :
        the alpha parameter of the variable
    theta :
        the theta parameter of the variable
    interval_low :
        the low interval of the variable
    order :
        the order of the model to calculate the orthogonal polynomials and norm 
        squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """

    __slots__ = ('alpha', 'theta')

    # This is the standardized form required for the UQPCE variable basis and
    # norm squared.
    equation = '(x**(A-1) * exp(-x)) / (A-1)!'

    def __init__(self, alpha, theta, interval_low=0, order=2, name='', number=0):

        if not ((alpha > 0) and (theta > 0)):
            raise VariableInputError(
                'GammaVariable alpha and theta must be greater than 0.'
            )

        self.alpha = alpha
        self.theta = theta
        self.interval_low = interval_low
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.GAMMA
        self.type = UncertaintyType.ALEATORY
        self.dist = gamma(self.alpha, scale=self.theta, loc=self.interval_low)

        low = 0
        high = 'oo'

        self.check_num_string()

        x = symbols(self.var_str)

        parsed_dist = parse_expr(
            self.equation,
            local_dict={'A':parse_expr(str(Fraction(self.alpha))), 'x':x}
        )

        self.recursive_var_basis(parsed_dist, low, high, self.order)
        self.norm_sq_vals = np.zeros(len(self.var_orthopoly_vect))
        self.create_norm_sq(low, high, parsed_dist)

        # if inf bounds, find approximate bound
        low_percent = 8e-17
        high_percent = 1 - low_percent
        
        self.low_approx = self.interval_low
        self.high_approx = self.dist.ppf(high_percent)

        upper = self.dist.ppf(high_percent)

        self.bounds = (self.interval_low, upper)
        self.std_bounds = (low, self.standardize_points(upper))

    def generate_samples(self, count, standardize=False):
        """
        Overrides the Variable class generate_samples to align with
        a gamma distribution.

        Parameters
        ----------
        count :
            the number of points needed to be generated
        standardize :
            boolean for generating samples for the standardized version of the
            variable
        """
        return super(ContinuousVariable, self).generate_samples(count, standardize=standardize)
    
    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with
        a gamma distribution.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        standard = (getattr(self, orig) - self.interval_low) / self.theta
        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        return (values - self.interval_low) / self.theta


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        return (value * self.theta) + self.interval_low


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with
        a gamma distribution.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """
        shift = 15

        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mx > shift) or (mn < 0):
            print(
                f'Large standardized value for variable {self.name} '
                'with gamma distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the
        values that might contain it.
        """
        if isinstance(self.alpha, str) and 'pi' in self.alpha:
            self.alpha = float(self.alpha.replace('pi', str(np.pi)))

        if isinstance(self.theta, str) and 'pi' in self.theta:
            self.theta = float(self.theta.replace('pi', str(np.pi)))

    def get_mean(self):
        """
        Return the mean of the variable.
        """
        return self.interval_low + (self.alpha * self.theta)

class LognormalVariable(ContinuousVariable):
    """
    Represents a lognormal variable. The methods in this class correspond to
    those of a lognormal variable.

    Parameters
    ----------
    mu :
        the mean, or expected value, of the variable
    stdev :
        the standard deviation of the variable's natural logarithm
    interval_low :
        the low interval of the variable
    order :
        the order of the model to calculate the orthogonal polynomials and norm 
        squared values
    name :
        the name of the variable
    number :
        the number of the variable from the file
    """

    __slots__ = ('mu', 'stdev')

    # This is the standardized form required for the UQPCE variable basis and
    # norm squared.
    equation = '(1/(x*sqrt(2*pi))) * exp(-(ln(x))**2/2)'

    def __init__(self, mu, stdev, interval_low=0, order=2, name='', number=0):

        if not (stdev > 0):
            raise VariableInputError(
                'LognormalVariable stdev must be greater than 0.'
            )

        self.mu = mu
        self.stdev = stdev
        self.interval_low = interval_low
        self.order = order
        
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)

        self.distribution = Distribution.LOGNORMAL
        self.type = UncertaintyType.ALEATORY
        self.dist = lognorm(
            s=self.stdev, scale=np.exp(self.mu), loc=self.interval_low
        )

        low = 0
        high = 'oo'

        self.check_num_string()

        x = symbols(self.var_str)

        parsed_dist = parse_expr(
            self.equation,
            local_dict={
                's':parse_expr(str(Fraction(self.stdev))), 
                'mu':parse_expr(str(Fraction(self.mu))), 'x':x}
        )

        self.recursive_var_basis(parsed_dist, low, high, self.order)
        self.norm_sq_vals = np.zeros(len(self.var_orthopoly_vect))
        self.create_norm_sq(low, high, parsed_dist)

        # if inf bounds, find approximate bound
        low_percent = 8e-17
        high_percent = 1 - low_percent
        
        self.low_approx = self.interval_low
        self.high_approx = self.dist.ppf(high_percent)

        upper = self.dist.ppf(high_percent)

        self.bounds = (self.interval_low, upper)
        self.std_bounds = (low, self.standardize_points(upper))

    def generate_samples(self, count, standardize=False):
        """
        Overrides the Variable class generate_samples to align with
        a lognormal distribution.

        Parameters
        ----------
        count :
            the number of points needed to be generated
        standardize :
            boolean for generating samples for the standardized version of the
            variable
        """
        return super(ContinuousVariable, self).generate_samples(
            count, standardize=standardize
        )
    
    def standardize(self, orig, std_vals):
        """
        Overrides the Variable class standardize to align with
        a lognormal distribution.

        Parameters
        ----------
        orig :
            the un-standardized values
        std_vals :
            the attribue name for the standardized vals
        """
        x_orig = getattr(self, orig)
        unstd_cdf = self.dist.cdf(x_orig)
        standard = lognorm(s=1, scale=1).ppf(unstd_cdf)

        setattr(self, std_vals, standard)

        return getattr(self, std_vals)


    def standardize_points(self, values):
        """
        Standardizes and returns the inputs points.

        Parameters
        ----------
        values :
            unstandardized points corresponding to the variable's distribution
        """
        unstd_cdf = self.dist.cdf(values)

        return lognorm(s=1, scale=1).ppf(unstd_cdf)


    def unstandardize_points(self, value):
        """
        Calculates and returns the unscaled variable value from the
        standardized value.

        Parameters
        ----------
        value :
            the standardized value to be unstandardized
        """
        std_cdf = lognorm(s=1, scale=1).cdf(value)

        return self.dist.ppf(std_cdf)


    def check_distribution(self, X):
        """
        Overrides the Variable class check_distribution to align with
        a lognormal distribution.

        Parameters
        ----------
        X :
           The array of samples to check against the variable distribution
        """
        shift = lognorm(s=1, scale=1).ppf(0.99)
        std_vals = self.standardize_points(X)

        mx = np.max(std_vals)
        mn = np.min(std_vals)

        if rank == 0 and (mx > shift) or (mn < 0):
            print(
                f'Large standardized value for variable {self.name} '
                'with lognormal distribution found. Check input and run matrix.', 
                file=sys.stderr
            )
            return -1


    def check_num_string(self):
        """
        Searches to replace sring 'pi' with its numpy equivalent in any of the
        values that might contain it.
        """
        if isinstance(self.mu, str) and 'pi' in self.mu:
            self.mu = float(self.mu.replace('pi', str(np.pi)))

        if isinstance(self.stdev, str) and 'pi' in self.stdev:
            self.stdev = float(self.stdev.replace('pi', str(np.pi)))


    def get_mean(self):
        """
        Return the mean of the variable.
        """
        return self.dist.stats('m')

class GaussianMixtureVariable(Variable):
    """
    Represents a Gaussian Mixture Model variable.
    
    A GMM is a weighted sum of multiple Gaussian distributions:
    p(x) = Σ w_i * N(x; μ_i, σ_i²)
    
    Parameters
    ----------
    weights : array-like
        Weights for each Gaussian component (will be normalized to sum to 1)
    means : array-like
        Means of each Gaussian component
    stdevs : array-like
        Standard deviations of each Gaussian component
    order : int
        Maximum polynomial order for PCE
    number : int
        Variable number/index
    name : str
        Variable name
    """
    
    __slots__ = ('weights', 'means', 'stdevs', 'n_components', 'mean', 'stdev', 
                 'is_standard_normal', '_pdf_cache')
    
    def __init__(self, weights, means, stdevs, order=2, name='', number=0):
        """Initialize a Gaussian Mixture Model variable."""
        
        # Validate and store GMM parameters
        self.weights = np.atleast_1d(weights).astype(float)
        self.means = np.atleast_1d(means).astype(float)
        self.stdevs = np.atleast_1d(stdevs).astype(float)
        self.n_components = len(self.weights)
        
        if not (len(self.weights) == len(self.means) == len(self.stdevs)):
            raise VariableInputError(
                'weights, means, and stdevs',
                'GaussianMixtureVariable weights, means, and stdevs must have the same length.'
            )
        
        if self.n_components == 0:
            raise VariableInputError(
                'components',
                'GaussianMixtureVariable must have at least one component.'
            )
        
        if not np.all(self.weights > 0):
            raise VariableInputError(
                'weights',
                'GaussianMixtureVariable weights must all be greater than 0.'
            )
        
        if not np.all(self.stdevs > 0):
            raise VariableInputError(
                'stdevs', 
                'GaussianMixtureVariable stdevs must all be greater than 0.'
            )
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if np.abs(weight_sum - 1.0) > 1e-10:
            self.weights = self.weights / weight_sum
            if rank == 0 and weight_sum != 1.0:
                print(
                    f'GaussianMixtureVariable: Normalized weights to sum to 1.0 '
                    f'(original sum was {weight_sum:.6f})', 
                    file=sys.stderr
                )
        
        # Calculate mixture statistics
        self.mean = float(np.sum(self.weights * self.means))
        e_x_squared = np.sum(self.weights * (self.stdevs**2 + self.means**2))
        gmm_var = e_x_squared - self.mean**2
        self.stdev = float(np.sqrt(gmm_var))
        
        # Set basic attributes required by Variable
        self.order = order
        self.name = f'x{number}' if name == '' else name
        self.var_str = f'x{number}'
        self.x = symbols(self.var_str)
        
        # Set distribution type
        self.distribution = Distribution.GAUSSIAN_MIXTURE
        self.type = UncertaintyType.ALEATORY
        
        # Calculate bounds (following existing UQPCE patterns)
        bounds_factor = 4
        gmm_low = float(np.min(self.means - bounds_factor * self.stdevs))
        gmm_high = float(np.max(self.means + bounds_factor * self.stdevs))
        
        # Use normal approximation for extended bounds
        self.dist = norm(loc=self.mean, scale=self.stdev)
        low_percent = 1e-7
        high_percent = 1 - low_percent
        norm_low = self.dist.ppf(low_percent)
        norm_high = self.dist.ppf(high_percent)
        
        self.interval_low = min(norm_low, gmm_low)
        self.interval_high = max(norm_high, gmm_high)
        self.low_approx = self.interval_low
        self.high_approx = self.interval_high
        self.bounds = (self.low_approx, self.high_approx)
        
        # Check if this is essentially a standard normal
        self.is_standard_normal = (
            self.n_components == 1 and 
            abs(self.mean) < 1e-10 and 
            abs(self.stdev - 1.0) < 1e-10
        )
        
        # Cache for PDF evaluations
        self._pdf_cache = {}
        
        # Generate orthogonal polynomials
        self.generate_orthopoly()
        
        # Set standardized bounds
        self.std_bounds = (
            self.standardize_points(self.low_approx),
            self.standardize_points(self.high_approx)
        )
        
        # Check for numerical issues
        self.check_num_string()
        self.check_distribution()
    
    def generate_orthopoly(self):
        """
        Generate orthogonal polynomials for the GMM.
        Uses Hermite for standard normal, Gram-Schmidt otherwise.
        """
        if self.is_standard_normal:
            self._generate_hermite_polynomials()
        else:
            self._generate_orthogonal_polynomials_gram_schmidt()
    
    def _generate_hermite_polynomials(self):
        """Generate standard Hermite polynomials for standard normal GMM."""
        from sympy import zeros
        
        self.var_orthopoly_vect = zeros(self.order + 1, 1)
        x = self.x
        
        # Physicists' Hermite polynomials: H_{n+1} = xH_n - nH_{n-1}
        for n in range(self.order + 1):
            if n == 0:
                self.var_orthopoly_vect[n] = 1
            elif n == 1:
                self.var_orthopoly_vect[n] = x
            else:
                self.var_orthopoly_vect[n] = expand(
                    x * self.var_orthopoly_vect[n - 1] - 
                    (n - 1) * self.var_orthopoly_vect[n - 2]
                )
        
        # Convert to numpy array (following UQPCE pattern)
        self.var_orthopoly_vect = np.array(self.var_orthopoly_vect).astype(object).T[0]
        
        # Use factorial norm squared for standard Hermite
        self.norm_sq_vals = np.array([math.factorial(i) for i in range(self.order + 1)])
        
        # Create norm_sq attribute for compatibility
        self.create_norm_sq()
    
    def _generate_orthogonal_polynomials_gram_schmidt(self):
        """
        Generate orthogonal polynomials using Gram-Schmidt process.
        Adapted from DiscreteVariable.recursive_var_basis() pattern.
        """
        # Initialize polynomial vector
        self.var_orthopoly_vect = np.zeros(self.order + 1, dtype=object)
        self.norm_sq_vals = np.zeros(self.order + 1)
        
        # Use recursive approach similar to DiscreteVariable
        self._recursive_orthogonalization(self.order)
        
        # Create norm_sq attribute for compatibility
        self.create_norm_sq()
    
    def _recursive_orthogonalization(self, order):
        """
        Recursively generate orthogonal polynomials up to given order.
        Following the pattern from DiscreteVariable.recursive_var_basis().
        """
        if order == 0:
            # Base case: P_0 = 1
            self.var_orthopoly_vect[0] = 1
            self.norm_sq_vals[0] = self._compute_inner_product(1, 1)
            return
        
        # Recursive case: build all polynomials up to 'order'
        self._recursive_orthogonalization(order - 1)
        
        # Start with monomial x^order
        x = self.x
        current_poly = x**order
        
        # Gram-Schmidt: subtract projections onto all previous polynomials
        for i in range(order):
            prev_poly = self.var_orthopoly_vect[i]
            
            # CRITICAL FIX: Compute projection using current_poly, not x**order
            # This ensures we're projecting the partially orthogonalized polynomial
            numerator = self._compute_inner_product(current_poly, prev_poly)
            denominator = self.norm_sq_vals[i]
            
            if abs(denominator) < 1e-10:
                print(f"Warning: Near-zero norm squared for polynomial {i}", file=sys.stderr)
                continue
                
            projection_coeff = numerator / denominator
            
            # Subtract projection
            current_poly = current_poly - projection_coeff * prev_poly
        
        # Simplify and store
        current_poly = expand(current_poly)
        self.var_orthopoly_vect[order] = current_poly
        self.norm_sq_vals[order] = self._compute_inner_product(current_poly, current_poly)
        
        # Check for numerical issues (following UQPCE pattern)
        if order == self.order and (np.array(self.var_orthopoly_vect) == 0).any():
            print(
                f'Variable {self.name} has at least one orthogonal polynomial '
                f'that is zero. The model may not be accurate', 
                file=sys.stderr
            )
    
    def _compute_inner_product(self, poly1, poly2):
        """
        Compute inner product <poly1, poly2> with respect to standardized GMM weight.
        """
        x = self.x
        
        # Convert symbolic polynomials to callable functions
        if hasattr(poly1, 'free_symbols'):
            poly1_func = lambdify(x, poly1, 'numpy')
        else:
            poly1_func = lambda x_val: float(poly1) * np.ones_like(x_val)
        
        if hasattr(poly2, 'free_symbols'):
            poly2_func = lambdify(x, poly2, 'numpy')
        else:
            poly2_func = lambda x_val: float(poly2) * np.ones_like(x_val)
        
        def integrand(x_val):
            p1 = poly1_func(x_val)
            p2 = poly2_func(x_val)
            pdf = self._gmm_pdf_standardized(x_val)
            return p1 * p2 * pdf
        
        # Integrate with appropriate bounds
        result, error = quad(integrand, -10, 10, limit=200, epsabs=1e-12, epsrel=1e-12)
        
        # Warn if integration error is large
        if abs(error) > 1e-6 * abs(result) and rank == 0:
            print(
                f'Warning: Large integration error in inner product computation '
                f'for variable {self.name}: {error:.2e}',
                file=sys.stderr
            )
        
        return result
    
    def _gmm_pdf_standardized(self, x_std):
        """PDF of the standardized GMM (mean=0, std=1)."""
        # Transform back to original space
        x_raw = self.mean + x_std * self.stdev
        
        # Compute GMM PDF in original space
        pdf = 0.0
        for w, m, s in zip(self.weights, self.means, self.stdevs):
            pdf += w * np.exp(-(x_raw - m)**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
        
        # Apply Jacobian for the transformation to standardized space
        return pdf * self.stdev
    
    def create_norm_sq(self):
        """
        Create norm_sq attribute for compatibility with UQPCE.
        Following the pattern from other Variable classes.
        """
        # norm_sq_vals already computed during orthogonalization
        # This method exists for API compatibility
        pass
    
    def get_norm_sq_val(self, matrix_val):
        """
        Returns the norm squared value corresponding to the matrix value.
        Following Variable class interface.
        """
        if matrix_val < len(self.norm_sq_vals):
            return float(self.norm_sq_vals[matrix_val])
        else:
            raise ValueError(f"Norm squared value not computed for index {matrix_val}")
    
    def generate_samples(self, count, standardize=False):
        """
        Generate samples from the GMM distribution.
        Following Variable class interface.
        """
        # Select components based on weights
        components = np.random.choice(self.n_components, size=count, p=self.weights)
        samples = np.zeros(count)
        
        # Generate samples from each component
        for i in range(self.n_components):
            mask = components == i
            n_samples = np.sum(mask)
            if n_samples > 0:
                samples[mask] = np.random.normal(
                    self.means[i], 
                    self.stdevs[i], 
                    n_samples
                )
        
        if standardize:
            return self.standardize_points(samples)
        
        return samples
    
    def cdf_sample(self, cdf_vals):
        """
        Sample from the GMM using CDF values.
        This is approximate using the overall GMM's normal approximation.
        """
        # Use normal approximation for CDF sampling
        # This is consistent with how UQPCE handles complex distributions
        return self.dist.ppf(cdf_vals)
    
    def resample(self, count):
        """
        Generate resampled values for PCE internal use.
        Following Variable class pattern.
        """
        samps = self.generate_samples(count, standardize=True)

        return samps
    
    def standardize(self, orig, std_vals):
        """
        Standardize attribute values.
        Following Variable class interface.
        """
        original = getattr(self, orig)
        standardized = self.standardize_points(original)
        setattr(self, std_vals, standardized)
        return getattr(self, std_vals)
    
    def standardize_points(self, values):
        """
        Transform points to standardized space (mean=0, std=1).
        """
        return (np.atleast_1d(values) - self.mean) / self.stdev
    
    def unstandardize_points(self, values):
        """
        Transform points from standardized space to original space.
        """
        return np.atleast_1d(values) * self.stdev + self.mean
    
    def check_distribution(self, check_var='vals'):
        """
        Check if the distribution is reasonable.
        Following Variable class pattern.
        """
        if not hasattr(self, check_var):
            return
        
        vals = getattr(self, check_var)
        std_vals = self.standardize_points(vals)
        
        mx = np.max(std_vals)
        mn = np.min(std_vals)
        
        if rank == 0 and (mx > 4.5 or mn < -4.5):
            print(
                f'Large standardized value for variable {self.name} '
                f'with Gaussian mixture distribution found. '
                f'Max: {mx:.2f}, Min: {mn:.2f}. Check input and run matrix.',
                file=sys.stderr
            )
    
    def check_num_string(self):
        """
        Check for string representations of numbers.
        Following Variable class pattern.
        """
        # GMM doesn't use string inputs, but method exists for compatibility
        pass
    
    def get_mean(self):
        """Return the mean of the GMM."""
        return self.mean
    
    def get_variance(self):
        """Return the variance of the GMM."""
        return self.stdev**2
    
    def get_stdev(self):
        """Return the standard deviation of the GMM."""
        return self.stdev

class EpistemicVariable(UniformVariable):

    
    def __init__(self, interval_low, interval_high, **kwargs):
        super().__init__(interval_low, interval_high, **kwargs)
        self.type = UncertaintyType.EPISTEMIC
