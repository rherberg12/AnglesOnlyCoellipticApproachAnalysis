"""
Analysis Script for Aerospace Corporation Technical Interview
Ryan Herberg; May 4, 2026

Content:
Includes investigation of camera parameter definition and resultant performance from coelliptic
approach and applied observability maneuvers. Confirms optimal maneuver to attain observability
requirements for angles only navigation, then compares camera resolution and parameters for
achieved angular measurement and resultant range uncertainty, finally derives required maneuver
to attain similar performance from each camera definition.

Source:
Leverages heavily the work accomplished by David Woffinden in his Utah State PhD Thesis entitled:
"Angles-Only Navigation for Autonomous Orbital Rendezvous"
LINK - https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1011&context=etd
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
 
# =============================================================================
# CONSTANTS
# =============================================================================
MU_EARTH   = 3.986004418e14      # m^3/s^2
R_EARTH    = 6.3781e6            # m
J2         = 1.08263e-3          # J2 coefficient
P_SR       = 4.57e-6             # N/m^2  solar radiation pressure at 1 AU
C_R        = 1.3                 # reflectivity coefficient (assumed)
GEO_RADIUS = 42_164_169.0        # m  (GEO semi-major axis)
GEO_ALT    = GEO_RADIUS - R_EARTH
 
@dataclass # used to automatically define __init__ and other boilerplate for simple attribute definition
class OrbitParams:
    a: float = GEO_RADIUS          # sma [m]
    inc: float = np.radians(0.0)   # inclination [rad]
    ecc: float = 0.0               # eccentricity
 
    @property
    def n(self):
        return np.sqrt(MU_EARTH / self.a**3)
 
    @property
    def T(self):
        return 2 * np.pi / self.n
 
 
# =============================================================================
# DYNAMICS — CW PROPAGATOR
# =============================================================================
 
class CWPropagator:
    def __init__(self, orbit: OrbitParams): # define self.orbit as the dataclass with properties above
        self.orbit = orbit
        self.n = orbit.n
 
    def stm(self, dTau):
        """
        CW state transition matrix definition
        RAC coordinate frame, Radial (pos. out), Along track, cross track

        dTau: rad, mean motion * dt
        """

        dT = dTau # n * dt
        s  = np.sin(dT)
        c  = np.cos(dT)
        n = self.n
 
        # Phi_rr (3x3)
        Phi_rr = np.array([
            [4 - 3*c, 0, 0],
            [6*(s - dT), 1, 0],
            [0, 0, c],
        ])
 
        # Phi_rv (3x3)
        Phi_rv = np.array([
            [s/n, 2*(1 - c)/n,   0],
            [2*(c - 1)/n, (4*s - 3*dT)/n, 0],
            [0, 0, s/n,]
        ])
 
        # Phi_vr (3x3)
        Phi_vr = np.array([
            [3*n*s, 0, 0],
            [6*n*(c - 1), 0, 0],
            [0, 0, -n*s]
        ])
 
        # Phi_vv (3x3)
        Phi_vv = np.array([
            [c, 2*s, 0],
            [-2*s, 4*c - 3, 0],
            [0, 0, c]
        ])
 
        Phi = np.zeros((6, 6))
        Phi[0:3, 0:3] = Phi_rr
        Phi[0:3, 3:6] = Phi_rv
        Phi[3:6, 0:3] = Phi_vr
        Phi[3:6, 3:6] = Phi_vv
 
        return Phi
 
    def stm_blocks(self, dTheta):
        Phi = self.stm(dTheta)
        return Phi[0:3,0:3], Phi[0:3,3:6], Phi[3:6,0:3], Phi[3:6,3:6]
 
    def propagate(self, x0, t_arr):
        states = np.zeros((len(t_arr), 6))
        x0_nd = x0.copy()
 
        tau0 = self.n * t_arr[0]
        for i, t in enumerate(t_arr):
            tau = self.n * t
            Phi = self.stm(tau - tau0)
            x_nd = Phi @ x0_nd
            states[i] = x_nd
 
        return states
 
    def delta_r_impulsive(self, dv, t_burn, t_arr):
        """
        Compute resultant change in position from dV applied.
        """
        delta_r = np.zeros((len(t_arr), 3))
 
        for i, t in enumerate(t_arr):
            if t < t_burn:
                delta_r[i] = 0.0
            else:
                tau_diff = self.n * (t - t_burn)
                _, Phi_rv, _, Phi_vv = self.stm_blocks(tau_diff)
                delta_r[i] = Phi_rv @ dv
                # delta_v[i] = Phi_vv @ dv
 
        return delta_r
 
class PerturbationModel:
    """
    First-order analytical perturbation accelerations on the relative state.
    
    J2 & SRP for GEO
    """
 
    def __init__(self, orbit: OrbitParams, Am_chaser, Am_target, sun_hat0 = None):
        """
        Args:
            orbit:      target orbit parameters
            Am_chaser:  chaser area-to-mass ratio [m^2/kg]
            Am_target:  target area-to-mass ratio [m^2/kg]
            sun_hat:    unit vector to sun in LVLH (default: +x along-track)
        """
        self.orbit    = orbit
        self.n        = orbit.n
        self.Am_diff  = Am_chaser - Am_target   # differential A/m
        self.sun_hat0  = sun_hat0 if sun_hat0 is not None else np.array([1.0, 0.0, 0.0])

    def j2_differential_acceleration(self, delta_r_RAC):
        """
        https://www.sciencedirect.com/science/article/pii/S2090997713000333
        """
        a  = self.orbit.a
        Re = R_EARTH
        mu = MU_EARTH

        # Gradient of J2 radial acceleration w.r.t. radial coordinate at equator
        # a_J2_R = (3/2)*J2*mu*Re^2 / r^4  (outward, equatorial x[radial]/r^5 -> / r^4)
        # d(a_J2_R)/dR = -6*J2*mu*Re^2 / a^5
        grad = -6.0 * J2 * MU_EARTH * Re**2 / a**5   # 1/s^2

        dR = delta_r_RAC[0]   # radial component of relative position


        """Inertial formulation for reference:
        c = J2*mu*Re**2/2
        r = np.linalg.norm(x)

        a_j2_x = 3*c*(x[0]/r**5)*(1-5*(x[2]/r)**2)
        a_j2_y = 3*c*(x[1]/r**5)*(1-5*(x[2]/r)**2)
        a_j2_z = 3*c*(x[2]/r**5)*(3-5*(x[2]/r)**2)
        """

        # Only radial gradient is significant at equatorial GEO
        # Along-track and cross-track differential J2 are second-order
        return np.array([grad * dR, 0.0, 0.0])

    def sun_angle(self, t): # rudimentary sun angle progression, planar rotation
        dTau = self.n * t
        R3 = np.array([[np.cos(dTau), -np.sin(dTau),0],
                      [np.sin(dTau),np.cos(dTau),0],
                      [0,0,1]])
        sun_hat = R3 @ self.sun_hat0 # transpose because in RAC frame its rotating "backwords"
        return sun_hat
 
    def srp_differential_acceleration(self, sun_hat):
        a_srp = -C_R * P_SR * self.Am_diff * sun_hat
        return a_srp
 
    def delta_r_perturbations(self, x0, t_arr):
        prop = CWPropagator(self.orbit)
        n    = self.n
        dt   = t_arr[1] - t_arr[0]
 
        delta_r = np.zeros((len(t_arr), 3))
 
        # Propagate nominal for J2 reference positions
        r_nom = prop.propagate(x0, t_arr)[:, :3]
 
        for i, t in enumerate(t_arr):
            tau = n * t
            accum = np.zeros(3)
 
            for j in range(i):
                t_mu   = t_arr[j]
                tau_mu = n * t_mu
                tau_diff = tau - tau_mu
 
                _, Phi_rv, _, Phi_vv = prop.stm_blocks(tau_diff)
 
                # J2 at nominal position at t_mu
                a_j2 = self.j2_differential_acceleration(r_nom[j])
                # SRP (constant)
                a_srp = self.srp_differential_acceleration(self.sun_angle(t_mu))
 
                a_total = a_j2 + a_srp   # m/s^2

                accum += Phi_rv @ (a_total * dt) # necessary to redefine what current velocity change is by recursively calculating for each time.
                # v_accum += Phi_vv @ (a_total * dt)
 
            delta_r[i] = accum
 
        return delta_r
 
 
# =============================================================================
# CAMERA MODEL
# =============================================================================
 
@dataclass
class Camera:
    """
    Camera sensor model for angular measurement noise.
 
    FOV is fixed; focal length and IFOV scale with resolution.
    IFOV (= epsilon) is the angular measurement accuracy per pixel.
    """
    megapixels: float # sensor resolution [Mpx]
    fov_deg: float = 10.0     # full FOV, square sensor [deg]
    pixel_pitch_um: float = 5.5  # pixel pitch [μm]
 
    @property
    def fov_rad(self):
        return np.radians(self.fov_deg)
 
    @property
    def n_pixels(self):
        """Pixels per side (4:3) and use longest side."""
        return np.sqrt(self.megapixels * 1e6 / (4*3)) * 4
 
    @property
    def focal_length_mm(self):
        """Focal length [mm] to achieve fixed FOV with given pitch & Npx."""
        p_mm = self.pixel_pitch_um * 1e-3
        return (self.n_pixels * p_mm) / (2.0 * np.tan(self.fov_rad / 2.0))
 
    @property
    def ifov_rad(self):
        """Instantaneous FOV per pixel [rad] = epsilon in Woffinden."""
        p_m = self.pixel_pitch_um * 1e-6
        f_m = self.focal_length_mm * 1e-3
        return p_m / f_m
 
    @property
    def ifov_urad(self):
        return self.ifov_rad * 1e6
 
    @property
    def ifov_deg(self):
        return np.degrees(self.ifov_rad)
 
    def summary(self):
        return (f"{self.megapixels:.0f} Mpx | "
                f"N={self.n_pixels:.0f} px | "
                f"f={self.focal_length_mm:.1f} mm | "
                f"IFOV={self.ifov_urad:.2f} μrad/px")
 
 
# =============================================================================
# LOS MEASUREMENT MODEL
# =============================================================================
 
class LOSMeasurement:
    """
    Line-of-sight unit vector measurement model.

    Azimuth az:   angle in the y-z plane from z-axis (RAC)
    Elevation el:     angle from y-z plane toward x (RAC)
    LOS unit vector: i_r = [sin(e), cos(e)sin(a), cos(e)cos(a)]
    """
 
    @staticmethod
    def los_vector(r):
        mag = np.linalg.norm(r)
        if mag < 1e-10:
            return np.zeros(3)
        return r / mag
 
    @staticmethod
    def azimuth_elevation(r):
        x, y, z = r
        az = np.arctan2(y, z)
        el     = np.arctan2(x, np.sqrt(y**2 + z**2))
        return az, el
 
    @staticmethod
    def los_from_angles(az, el):
        """Reconstruct LOS vector from azimuth and elevation"""
        return np.array([
            np.sin(el),
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
        ])
 
 
# =============================================================================
# OBSERVABILITY METRICS — Woffinden Ch. 8
# =============================================================================
 
class ObservabilityMetrics:
    """
    Angle between nominal los and new los
    """
 
    @staticmethod
    def observability_angle(r_nom,r_true):
        i_nom  = LOSMeasurement.los_vector(r_nom)
        i_true = LOSMeasurement.los_vector(r_true)
        dot    = np.clip(np.dot(i_nom, i_true), -1.0, 1.0)
        return np.arccos(dot)
 
    @staticmethod
    def perturbation_angle(r_nom, delta_r):
        """
        Angle between los from chaser to target and the chaser perturbed motion vector
        """
        mag_dr = np.linalg.norm(delta_r)
        if mag_dr < 1e-10:
            return 0.0
        i_nom = LOSMeasurement.los_vector(r_nom)
        dot   = np.clip(np.dot(-i_nom, delta_r / mag_dr), -1.0, 1.0)
        return np.arccos(dot)
 
    @staticmethod
    def detectability_range_error(r_nom, r_true, epsilon):
        """
        delta_rho ≈ epsilon * |r| / sin(theta)   Eq. 8.52
        """
        theta = ObservabilityMetrics.observability_angle(r_nom, r_true)
        r_mag = np.linalg.norm(r_nom)
        sin_t = np.sin(theta)
        if abs(sin_t) < 1e-12:
            return np.inf
        return epsilon * r_mag / sin_t
 
    @staticmethod
    def detectability_pct_range_error(r_nom, r_true, epsilon):
        """
        delta_rho / r ≈ epsilon / sin(theta)   Eq. 8.53
        """
        theta = ObservabilityMetrics.observability_angle(r_nom, r_true)
        sin_t = np.sin(theta)
        if abs(sin_t) < 1e-12:
            return np.inf
        return epsilon / sin_t
 
    @classmethod
    def compute_time_history(cls, r_nom_hist, r_true_hist, epsilon):
        """
        observability metrics computer over time history
        """
        N      = len(r_nom_hist)
        theta  = np.zeros(N)
        drho   = np.zeros(N)
        pct    = np.zeros(N)
 
        for i in range(N):
            theta[i] = cls.observability_angle(r_nom_hist[i], r_true_hist[i])
            drho[i]  = cls.detectability_range_error(
                           r_nom_hist[i], r_true_hist[i], epsilon)
            pct[i]   = cls.detectability_pct_range_error(
                           r_nom_hist[i], r_true_hist[i], epsilon)
 
        return {
            'theta':           np.degrees(theta),   # deg
            'delta_rho':       drho,                # m
            'pct_range_error': pct * 100.0          # percent
        }
 
 
# =============================================================================
# SCENARIO
# =============================================================================
 
@dataclass
class ManeuverDef:
    """Definition of a single impulsive maneuver."""
    dv: np.ndarray          # [dvx, dvy, dvz] m/s  in RAC
    t_burn: float           # time of maneuver [s] from scenario start
    label: str = ""
 
@dataclass
class ScenarioConfig:
    orbit:       OrbitParams = field(default_factory=OrbitParams)
    t_window:    float = 4 * 3600.0
    dt:          float = 30.0
    Am_chaser:   float = 0.02
    Am_target:   float = 0.02
    maneuver:    Optional[ManeuverDef] = None
    include_perturbations: bool = True
    dR:          float = -10_000.0
    sun_hat0:    np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self):
        n = self.orbit.n
        self.t_span = 2*np.pi/n + self.t_window   # time for chaser to reach r-bar = 0
        drift_rate = -1.5 * n * self.dR   # m/s, use x_dd term of CW and solve for x_dd=0, y_d? secular drift
        y0 = -drift_rate * self.t_span     # negative (chaser starts behind)

        # Coelliptic ICs: no radial velocity, no along-track velocity
        # CW will propagate the natural drift from the energy difference
        self.x0 = np.array([self.dR, y0, 0.0,
                            0.0,    drift_rate, 0.0])
        
        self.t_span = 2*np.pi/n # oveerride for one orbit cycle...
 
class Scenario:
    """
    Runs the CW propagation (nominal + maneuver) and computes
    the resulting position histories for observability analysis.
    """
 
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.prop   = CWPropagator(config.orbit)
        self.t_arr  = np.arange(0.0, config.t_span + config.dt, config.dt)
        self._run()
 
    def _run(self):
        cfg = self.config
        prop = self.prop
 
        self.r_nom = prop.propagate(cfg.x0, self.t_arr)[:, :3]
 
        if cfg.include_perturbations:
            pert_model = PerturbationModel(
                cfg.orbit, cfg.Am_chaser, cfg.Am_target, cfg.sun_hat0
            )
            self.dr_pert = pert_model.delta_r_perturbations(cfg.x0, self.t_arr)
        else:
            self.dr_pert = np.zeros_like(self.r_nom)
 
        if cfg.maneuver is not None:
            m = cfg.maneuver
            self.dr_maneuver = prop.delta_r_impulsive(
                m.dv, m.t_burn, self.t_arr
            )
        else:
            self.dr_maneuver = np.zeros_like(self.r_nom)
 
        self.dr_total = self.dr_pert + self.dr_maneuver
        self.r_true   = self.r_nom + self.dr_total
 
    @property # recomputes automatically if there is a change to the class parameter
    def t_hours(self):
        return self.t_arr / 3600.0
 
    def range_history(self):
        return np.linalg.norm(self.r_nom, axis=1)
 
 
# =============================================================================
# STUDY 1 — MANEUVER PROFILE TRADE
# =============================================================================
 
def study1_maneuver_profile_trade(
        base_config: ScenarioConfig,
        camera: Camera,
        dv_mag,
        t_burn_s = 0.0   # burn at t = 0.0 hr
    ):
    """
    Evaluate maneuver profile and optimal method for increased observability at this range.
    """
    maneuver_types = {
        '+ Cross Track':  np.array([0.0, 0.0, 1.0]),
        '+ Altitude':  np.array([1.0, 0.0, 0.0]),
        '+ 45 deg':       np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
        '- Radial':  np.array([-1.0, 0.0, 0.0]),
        '- 45 deg':       np.array([-1.0, 0.0, 1.0]) / np.sqrt(2),
    }
 
    results = {}
    epsilon = camera.ifov_rad
 
    for mtype, dv_hat in maneuver_types.items():
        dv_vec = dv_hat * dv_mag

        cfg = ScenarioConfig( # configure this maneuver scenario
            orbit   = base_config.orbit,
            t_window= base_config.t_window,
            dR = base_config.dR,
            dt      = base_config.dt,
            Am_chaser = base_config.Am_chaser,
            Am_target = base_config.Am_target,
            maneuver = ManeuverDef(dv=dv_vec, t_burn=t_burn_s,
                                    label=f"{mtype} {dv_mag:.3f} m/s"),
            include_perturbations = base_config.include_perturbations,
        )

        scen = Scenario(cfg)
        metrics = ObservabilityMetrics.compute_time_history(
            scen.r_nom, scen.r_true, epsilon
        )
        metrics['t_hours']    = scen.t_hours
        metrics['r_nom']      = scen.r_nom
        metrics['r_true']     = scen.r_true
        metrics['dr_total']   = scen.dr_total

        results[mtype] = metrics
 
    return results
 
 
# =============================================================================
# STUDY 2 — CAMERA RESOLUTION TRADE
# =============================================================================
 
def study2_camera_resolution_trade(
        base_config: ScenarioConfig,
        cameras: List[Camera],
        best_maneuver: ManeuverDef
    ):
    """
    Compare detectability range error across camera resolutions
    for the best maneuver profile identified in Study 1.
    """
    results = {}
 
    cfg = ScenarioConfig(
        orbit   = base_config.orbit,
        dR = base_config.dR,
        t_window= base_config.t_window,
        dt      = base_config.dt,
        Am_chaser = base_config.Am_chaser,
        Am_target = base_config.Am_target,
        maneuver  = best_maneuver,
        include_perturbations = base_config.include_perturbations,
    )
    scen = Scenario(cfg)
 
    for cam in cameras:
        metrics = ObservabilityMetrics.compute_time_history(
            scen.r_nom, scen.r_true, cam.ifov_rad
        )
        metrics['t_hours'] = scen.t_hours
        metrics['camera']  = cam
        results[cam.megapixels] = metrics
 
    return results, scen
 
 
# =============================================================================
# STUDY 3 — MANEUVER SIZING FOR EQUIVALENT PERFORMANCE
# =============================================================================
 
def study3_maneuver_sizing(
        base_config: ScenarioConfig,
        cameras: List[Camera],
        maneuver_type: np.ndarray,
        t_burn_s: float,
        target_pct_error: float = 1.0,   # target % range error
        dv_sweep: Optional[np.ndarray] = None
    ):
    """
    Derive dV required for each camera to achieve the same performance
    """
    if dv_sweep is None:
        dv_sweep = np.logspace(-3, 1, 80)   # 0.001 to 10 m/s
 
    results = {}
 
    for cam in cameras:
        epsilon = cam.ifov_rad
        theta_req_rad = np.arcsin(epsilon * 100.0 / target_pct_error) # required maneuver angle for camera resolution
 
        theta_req_deg = np.degrees(theta_req_rad)
        max_theta_arr = np.zeros(len(dv_sweep))
 
        for i, dv_mag in enumerate(dv_sweep):
            dv_vec = maneuver_type * dv_mag
            cfg = ScenarioConfig(
                orbit   = base_config.orbit,
                dR = base_config.dR,
                t_window= base_config.t_window,
                dt      = base_config.dt,
                Am_chaser = base_config.Am_chaser,
                Am_target = base_config.Am_target,
                maneuver  = ManeuverDef(dv=dv_vec, t_burn=t_burn_s),
                include_perturbations = base_config.include_perturbations,
            )
            scen = Scenario(cfg)
            metrics = ObservabilityMetrics.compute_time_history(
                scen.r_nom, scen.r_true, epsilon
            )
            max_theta_arr[i] = np.nanmax(metrics['theta'])
 
        # Find minimum ΔV that achieves theta_req
        above = max_theta_arr >= theta_req_deg
        if np.any(above):
            dv_req = dv_sweep[np.argmax(above)]
            achievable = True
            dv_vec = maneuver_type * dv_req
            cfg = ScenarioConfig( # configure this maneuver scenario
                orbit   = base_config.orbit,
                t_window= base_config.t_window,
                dR = base_config.dR,
                dt      = base_config.dt,
                Am_chaser = base_config.Am_chaser,
                Am_target = base_config.Am_target,
                maneuver = ManeuverDef(dv=dv_vec, t_burn=t_burn_s,
                                        label=f"{cam}, {dv_mag:.3f} m/s"),
                include_perturbations = base_config.include_perturbations,
            )

            scen = Scenario(cfg)
            metrics = ObservabilityMetrics.compute_time_history(
                scen.r_nom, scen.r_true, epsilon
            )
            metrics['t_hours']    = scen.t_hours
            metrics['r_nom']      = scen.r_nom
            metrics['r_true']     = scen.r_true
            metrics['dr_total']   = scen.dr_total
        else:
            dv_req = np.nan
            achievable = False
 
        results[cam.megapixels] = {
            'dv_required':      dv_req,
            'theta_required_deg': theta_req_deg,
            'achievable':       achievable,
            'camera':           cam,
            'dv_sweep':         dv_sweep,
            'max_theta_sweep':  max_theta_arr,
            'maneuver': metrics,
        }
 
    return results
 
 
# =============================================================================
# PLOTTING
# =============================================================================
 
# --- Style ---
COLORS = {
    '+ Cross Track': '#2196F3',   # blue
    '+ Altitude': '#FF5722',   # orange-red
    '+ 45 deg':      '#4CAF50',   # green
    '- Radial':     "#FF00DD",   # pink
    '- 45 deg':     "#FFF700",   # yellow
}
CAM_COLORS = ['#7C4DFF', '#00BCD4', '#FF9800', '#E91E63']  # 1,5,10,15 Mpx
STYLE = {'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
         'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
         'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
         'grid.color': '#21262d', 'grid.linewidth': 0.8,
         'text.color': '#e6edf3', 'legend.facecolor': '#1c2128',
         'legend.edgecolor': '#30363d'}
 
 
def apply_style():
    plt.rcParams.update(STYLE)
    plt.rcParams['font.family'] = 'monospace' 
 
def plot_study1(results: dict, dv_mag: float, save: bool = True):
    apply_style()
    maneuver_types = ['+ Cross Track', '+ Altitude', '+ 45 deg', '- Radial', '- 45 deg']
 
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'STUDY 1 — Maneuver Profile Trade  |  ΔV = {dv_mag:.3f} m/s',
                    fontsize=13, fontweight='bold', color='#e6edf3')
    gs = gridspec.GridSpec(3, 1, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for mtype in maneuver_types:
        key = mtype
        r = results[key]
        t = r['t_hours']
        c = COLORS[mtype]

        ax1.plot(t, r['theta'],           color=c, lw=1.8, label=mtype)
        ax2.plot(t, r['delta_rho'],  color=c, lw=1.8, label=mtype)
        ax3.plot(t, r['pct_range_error'], color=c, lw=1.8, label=mtype)

    for ax, ylabel, title in [
        (ax1, 'θ [deg]',        'Observability Angle θ(t)'),
        (ax2, 'δρ [m]',        'Detectability Range Error δρ(t)'),
        (ax3, 'δρ/r [%]',       'Detectability % Range Error δρ/r(t)'),
    ]:
        ax.axhline(1.0 if ax == ax3 else 0, color='#30363d', lw=0.8)
        if ax == ax3:
            ax.axhline(1.0, color="#007618", ls=':', lw=1.2,
                        label='1% threshold')
            ax.set_ylim([0.0,100.0])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.5)

    ax3.set_xlabel('Time [hr]')

    apply_style() 
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'STUDY 1 — Trajectories  |  ΔV = {dv_mag:.3f} m/s',
                    fontsize=13, fontweight='bold', color='#e6edf3')
    gs = gridspec.GridSpec(1, 2, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for mtype in maneuver_types:
        key = mtype
        if key not in results:
            continue
        r = results[key]
        t = r['t_hours']
        c = COLORS[mtype]
        radial = r['r_true'][:,0]
        alongTrack = r['r_true'][:,1]
        crossTrack = r['r_true'][:,2]

        ax1.plot(alongTrack, radial, color=c, lw=1.8, label=mtype)
        ax2.plot(crossTrack, radial,  color=c, lw=1.8, label=mtype)

    for ax, xlabel, title in [
        (ax1, 'Along-Track [m]',        'Orbital Plane'),
        (ax2, 'Cross-Track [m]',        'Along-Track Perspective'),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.5)

    ax1.set_ylabel('Radial [m]')

    plt.tight_layout()
    plt.show() 
 
def plot_study2(results: dict, scen: Scenario, cameras: List[Camera],
                save: bool = True):
    apply_style()
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('STUDY 2 — Camera Resolution Trade',
                 fontsize=13, fontweight='bold', color='#e6edf3')
    gs = gridspec.GridSpec(3, 1, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
 
    for cam, color in zip(cameras, CAM_COLORS):
        mpx = cam.megapixels
        r   = results[mpx]
        t   = r['t_hours']
        lbl = f'{mpx:.0f} Mpx  (IFOV={cam.ifov_urad:.2f} μrad)'
 
        ax1.plot(t, r['theta'],           color=color, lw=1.8, label=lbl)
        ax2.plot(t, r['delta_rho'],  color=color, lw=1.8, label=lbl)
        ax3.plot(t, r['pct_range_error'], color=color, lw=1.8, label=lbl)
 
    for ax, ylabel, title in [
        (ax1, 'θ [deg]',   'Observability Angle θ(t)  [same for all cameras]'),
        (ax2, 'δρ [m]',   'Detectability Range Error δρ(t)'),
        (ax3, 'δρ/r [%]',  'Detectability % Range Error δρ/r(t)'),
    ]:
        if ax == ax3:
            ax.axhline(1.0, color='#FFD700', ls=':', lw=1.2, label='1% req.')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.5)
 
    ax3.set_xlabel('Time [hr]')

    plt.tight_layout()
    plt.show()
 
 
def plot_study3(results: dict, cameras: List[Camera],
                target_pct: float = 1.0, save: bool = False):
    apply_style()
    fig, ax_r = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle(f'STUDY 3 — Required ΔV for {target_pct:.0f}% Range Error',
                 fontsize=13, fontweight='bold', color='#e6edf3')
    
    dv_reqs = []
    labels  = []
    colors  = []
    for cam, color in zip(cameras, CAM_COLORS):
        r = results[cam.megapixels]
        dv_reqs.append(r['dv_required'] if r['achievable'] else np.nan)
        labels.append(f'{cam.megapixels:.0f} Mpx')
        colors.append(color)
 
    x_pos = np.arange(len(cameras))
    bars = ax_r.bar(x_pos, dv_reqs, color=colors, edgecolor='#30363d', linewidth=0.8)
    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(labels)
    ax_r.set_ylabel('Required ΔV [m/s]')
    ax_r.set_title(f'ΔV Required to Achieve {target_pct:.0f}% Range Error')
    ax_r.grid(True, alpha=0.4, axis='y')
    for bar, val in zip(bars, dv_reqs):
        if not np.isnan(val):
            ax_r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                      f'{val:.4f} m/s', ha='center', va='bottom', fontsize=9)
    
    apply_style() 
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'STUDY 3 — Trajectories',
                    fontsize=13, fontweight='bold', color='#e6edf3')
    gs = gridspec.GridSpec(1, 2, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for cam, color in zip(cameras, CAM_COLORS):
        r = results[cam.megapixels]['maneuver']
        t = r['t_hours']
        radial = r['r_true'][:,0]
        alongTrack = r['r_true'][:,1]
        crossTrack = r['r_true'][:,2]

        ax1.plot(alongTrack, crossTrack, color=color, lw=1.8, label=f'{cam.megapixels:.0f} Mpx')
        ax2.plot(radial, crossTrack,  color=color, lw=1.8, label=f'{cam.megapixels:.0f} Mpx')

    for ax, xlabel, title in [
        (ax1, 'Along-Track [m]',        'Radial Perspective'),
        (ax2, 'Radial [m]',        'Along-Track Perspective'),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.5)

    ax1.set_ylabel('Cross-Track [m]')
    
    plt.tight_layout()
    plt.show()
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    study1 = True
    study2 = True
    study3 = True


    orbit = OrbitParams()

    Am_chaser = 0.02 # 
    Am_target = 0.02 #
 
    # --- Base config ---
    base_cfg = ScenarioConfig(
        orbit    = orbit,
        dR       = -10_000.0,
        t_window = 4 * 3600.0,
        dt       = 30.0,
        Am_chaser = Am_chaser,
        Am_target = Am_target,
        include_perturbations = False,
    )
 
    # --- Cameras ---
    cameras = [
        Camera(megapixels=1,  fov_deg=5.0, pixel_pitch_um=5.5),
        Camera(megapixels=5,  fov_deg=5.0, pixel_pitch_um=5.5),
        Camera(megapixels=10, fov_deg=5.0, pixel_pitch_um=5.5),
        Camera(megapixels=15, fov_deg=5.0, pixel_pitch_um=5.5),
    ]
 
    # Mid-grade camera for Study 1 (5 Mpx)
    cam_study1 = cameras[1]
    dv_mag = 0.02
    t_burn  = 0.0
 
    if study1:
        print("\n--- Study 1: Maneuver Profile Trade ---")
           
        s1_results = study1_maneuver_profile_trade(
            base_cfg, cam_study1, dv_mag, t_burn_s=t_burn
        )
        plot_study1(s1_results,dv_mag,save=False)

    best_dv_hat = np.array([0.0,0.0,1.0]) # + Cross Track
    best_maneuver = ManeuverDef(dv=best_dv_hat * dv_mag,
                             t_burn=t_burn,
                             label="+ Cross Track")

    if study2:
        print("\n--- Study 2: Camera Resolution Trade ---")
        s2_results, s2_scen = study2_camera_resolution_trade(
            base_cfg, cameras, best_maneuver
        )
        plot_study2(s2_results, s2_scen, cameras)
 
    if study3:
        print("\n--- Study 3: Maneuver Sizing for Equivalent Performance ---")
        target_pct = 1.0  # % range error requirement
        dv_sweep   = np.logspace(-3, 1, 50)
    
        s3_results = study3_maneuver_sizing(
            base_cfg, cameras,
            maneuver_type = best_dv_hat,
            t_burn_s      = t_burn,
            target_pct_error = target_pct,
            dv_sweep      = dv_sweep,
        )
        plot_study3(s3_results, cameras, target_pct)
 
if __name__ == '__main__':
    main()