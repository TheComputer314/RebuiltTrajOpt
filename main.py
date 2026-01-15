"""
FRC 2022 shooter trajectory optimization.

This program uses the Sleipnir NLP solver to find the initial pitch and yaw for a game
piece to hit the 2026 FRC game's target given an initial velocity.

This optimization problem formulation uses direct transcription of the flight dynamics, including
air resistance.

Based on the 2022 trajectory optimization example code by Tyler Veness.
https://github.com/SleipnirGroup/Sleipnir/blob/main/examples/frc_2022_shooter/main.py
"""

import math
from io import TextIOWrapper

import numpy as np
from numpy.linalg import norm
from sleipnir.autodiff import VariableMatrix, atan2, hypot
from sleipnir.optimization import ExitStatus, Problem

# Field dimensions
field_width = 8.043  # m
field_length = 16.518  # m
target_wrt_field = np.array(
    [
        [0],
        [0],
        [72 * 0.0254],
        [0.0],
        [0.0],
        [0.0],
    ]
)
# Physical characteristics
shooter_wrt_robot = np.array([[0.0], [0.0], [20 * 0.0254], [0.0], [0.0], [0.0]])
g = 9.81  # m/s²
max_shooter_velocity = 30  # m/s
ball_mass = 0.5 / 2.205  # kg
ball_diameter = 5.91 * 0.0254  # m

printResults = False


def lerp(a, b, t):
    return a + t * (b - a)


def f(x):
    """
    Apply the drag equation to a velocity.
    """
    # x' = x'
    # y' = y'
    # z' = z'
    # x" = −a_D(v_x)
    # y" = −a_D(v_y)
    # z" = −g − a_D(v_z)
    #
    # where a_D(v) = ½ρv² C_D A / m
    # (see https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation)
    rho = 1.204  # kg/m³
    C_D = 0.5
    m = ball_mass
    A = math.pi * ((ball_diameter / 2) ** 2)
    a_D = lambda v: 0.5 * rho * v**2 * C_D * A / m

    v_x = x[3, 0]
    v_y = x[4, 0]
    v_z = x[5, 0]
    return VariableMatrix(
        [[v_x], [v_y], [v_z], [-a_D(v_x)], [-a_D(v_y)], [-g - a_D(v_z)]]
    )


N = 40


def setup_problem(distance, robot_vx, robot_vy, max_horizontal_velocity):
    """
    Set up the problem and any shared constraints between the two solve modes (min and fix vel)
    """
    # Robot initial state
    robot_wrt_field = np.array([[-distance], [0], [0.0], [robot_vx], [robot_vy], [0.0]])

    shooter_wrt_field = robot_wrt_field + shooter_wrt_robot

    problem = Problem()

    # Set up duration decision variables
    T = problem.decision_variable()
    problem.subject_to(T >= 0)
    T.set_value(1)
    dt = T / N

    # Ball state in field frame
    #
    #     [x position]
    #     [y position]
    #     [z position]
    # x = [x velocity]
    #     [y velocity]
    #     [z velocity]
    X = problem.decision_variable(6, N)

    p = X[:3, :]

    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    v0_wrt_shooter = X[3:, :1] - shooter_wrt_field[3:, :]

    # Shooter initial position
    problem.subject_to(p[:, :1] == shooter_wrt_field[:3, :])

    # Dynamics constraints - RK4 integration
    h = dt
    for k in range(N - 1):
        x_k = X[:, k]
        x_k1 = X[:, k + 1]

        k1 = f(x_k)
        k2 = f(x_k + h / 2 * k1)
        k3 = f(x_k + h / 2 * k2)
        k4 = f(x_k + h * k3)
        problem.subject_to(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # Require final position is in center of target circle
    problem.subject_to(p[:, -1] == target_wrt_field[:3, :])

    # Require the final velocity is at least somewhat downwards by limiting horizontal velocity
    # and requiring negative vertical velocity
    problem.subject_to(hypot(v_x[-1], v_y[-1]) <= max_horizontal_velocity)
    problem.subject_to(v_z[-1] < 0)

    # Require the initial velocity is at least 45 degrees upwards
    # Shot angles shallower than 45 degrees tend to cause the solver to get stuck due to the
    # downwards velocity constraint above
    problem.subject_to(atan2(v_z[0], hypot(v_x[0], v_y[0])) >= np.deg2rad(45))

    return problem, shooter_wrt_field, v0_wrt_shooter, T, X


def min_velocity(distance, robot_vx, robot_vy):
    """
    Solve for minimum velocity.
    :returns: A tuple of [True, velocity, pitch, yaw, X] if it succeeds at a solve, and a tuple of[False, 0] if it fails.
    """
    problem, shooter_wrt_field, v0_wrt_shooter, T, X = setup_problem(
        distance, robot_vx, robot_vy, 6.5
    )

    p_x = X[0, :]
    p_y = X[1, :]
    p_z = X[2, :]

    v = X[3:, :]
    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    # Position initial guess is linear interpolation between start and end position
    for k in range(N):
        p_x[k].set_value(lerp(shooter_wrt_field[0, 0], target_wrt_field[0, 0], k / N))
        p_y[k].set_value(lerp(shooter_wrt_field[1, 0], target_wrt_field[1, 0], k / N))
        p_z[k].set_value(lerp(shooter_wrt_field[2, 0], target_wrt_field[2, 0], k / N))

    # Velocity initial guess is max initial velocity toward target
    uvec_shooter_to_target = target_wrt_field[:3, :] - shooter_wrt_field[:3, :]
    uvec_shooter_to_target /= norm(uvec_shooter_to_target)
    for k in range(N):
        v[:, k].set_value(
            shooter_wrt_field[3:, :] + max_shooter_velocity * uvec_shooter_to_target
        )

    # Require initial velocity is less than max shooter velocity
    #
    #   √(v_x² + v_y² + v_z²) = v
    #   v_x² + v_y² + v_z² = v²
    problem.subject_to(
        (v_x[0] - shooter_wrt_field[3, 0]) ** 2
        + (v_y[0] - shooter_wrt_field[4, 0]) ** 2
        + (v_z[0] - shooter_wrt_field[5, 0]) ** 2
        <= max_shooter_velocity**2
    )

    # Minimize initial velocity
    problem.minimize(v0_wrt_shooter.T @ v0_wrt_shooter)

    status = problem.solve(tolerance=0.01)
    if status == ExitStatus.SUCCESS:
        # Initial velocity vector with respect to shooter
        v0 = v0_wrt_shooter.value()
        velocity = norm(v0)
        pitch = math.atan2(v0[2, 0], math.hypot(v0[0, 0], v0[1, 0]))
        yaw = math.atan2(v0[1, 0], v0[0, 0])

        if printResults:
            print(f"Minimum velocity solve at distance {distance:.03f}:")
            print(f"Velocity = {velocity:.03f} m/s")
            print(f"Pitch = {np.rad2deg(pitch):.03f}°")
            print(f"Yaw = {np.rad2deg(yaw):.03f}°")

        return True, velocity, pitch, yaw, T.value(), X
    print(f"Infeasible at distance {distance:.03f} m with status {status.name}")
    return False, 0


def fixed_velocity(distance, robot_vx, robot_vy, target_vel, prev_X):
    """
    Solve for a fixed velocity.
    :param prev_X: The previous solve's state vectors, to act as an initial guess.
    :returns: A tuple of [True, velocity, pitch, yaw, X] if it succeeds at a solve, and a tuple of[False, 0] if it fails.
    """
    prev_v_x = prev_X[3, :]
    prev_v_y = prev_X[4, :]

    problem, shooter_wrt_field, v0_wrt_shooter, T, X = setup_problem(
        distance,
        robot_vx,
        robot_vy,
        # Horizontal velocity at target must be less than the previous solve's horizontal
        # velocity at target.
        # This makes the angle at which the shot goes in more steep, preventing the solver from
        # getting stuck with a shallow and infeasible shot angle
        math.hypot(prev_v_x[-1].value(), prev_v_y[-1].value()),
    )

    prev_p_x = prev_X[0, :]
    prev_p_y = prev_X[1, :]
    prev_p_z = prev_X[2, :]

    prev_v = prev_X[3:, :]

    p_x = X[0, :]
    p_y = X[1, :]
    p_z = X[2, :]

    v = X[3:, :]
    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    # Position initial guess is last solve's position
    for k in range(N):
        p_x[k].set_value(prev_p_x[k].value())
        p_y[k].set_value(prev_p_y[k].value())
        p_z[k].set_value(prev_p_z[k].value())

    # Velocity initial guess is last solve's velocity
    for k in range(N):
        v[:, k].set_value(prev_v[:, k].value())

    # Require initial velocity is equal to target
    #
    #   √(v_x² + v_y² + v_z²) = v
    #   v_x² + v_y² + v_z² = v²
    problem.subject_to(
        (v_x[0] - shooter_wrt_field[3, 0]) ** 2
        + (v_y[0] - shooter_wrt_field[4, 0]) ** 2
        + (v_z[0] - shooter_wrt_field[5, 0]) ** 2
        == target_vel**2
    )

    status = problem.solve(tolerance=0.01)
    if status == ExitStatus.SUCCESS:
        # Initial velocity vector with respect to shooter
        v0 = v0_wrt_shooter.value()
        velocity = norm(v0)
        pitch = math.atan2(v0[2, 0], math.hypot(v0[0, 0], v0[1, 0]))
        yaw = math.atan2(v0[1, 0], v0[0, 0])

        if printResults:
            print(f"Fixed velocity solve at distance {distance:.03f}:")
            print(f"Velocity = {velocity:.03f} m/s")
            print(f"Pitch = {np.rad2deg(pitch):.03f}°")
            print(f"Yaw = {np.rad2deg(yaw):.03f}°")

        return True, velocity, pitch, yaw, T.value(), X
    print(
        f"Infeasible at distance {distance:.03f} with velocity {target_vel:.03f} m/s with status {status.name}"
    )
    return False, 0


if __name__ == "__main__":
    file: TextIOWrapper = open("HubShotTable.java", "w")
    file.write("package frc.cotc.shooter;\n\n")

    file.write("import static java.util.Map.entry;\n\n")

    file.write("import edu.wpi.first.math.MathUtil;\n")
    file.write(
        "import edu.wpi.first.math.interpolation.InterpolatingTreeMap;\n"
    )
    file.write("import frc.cotc.shooter.ShotTable.ShotResult;\n")
    file.write("import java.util.Map;\n\n")

    file.write("public final class HubShotTable {\n")
    file.write("  private HubShotTable() {}\n\n")
    file.write("  private static final ShotTable table = new ShotTable();\n\n")
    file.write("  static {\n")

    start_distance = 0.5
    end_distance = 6.5

    distance_samples = 20
    velocity_samples = 20
    for i in range(0, distance_samples):
        distance = lerp(start_distance, end_distance, i / (distance_samples - 1))
        vx = 0
        vy = 0

        # Solve for minimum velocity
        min_vel_solve = min_velocity(distance, vx, vy)
        # If the position is possible, lerp between min velocity and max velocity
        # to search the in between velocities
        if min_vel_solve[0]:
            file.write("    table.put(\n")
            file.write(f"      {distance},\n")
            file.write("      makeTable(\n")
            file.write(
                f"        entry({min_vel_solve[1]}, new ShotResult("
                f"{np.rad2deg(min_vel_solve[2])},  {min_vel_solve[3]}, {min_vel_solve[4]})),\n"
            )
            prev_solve = min_vel_solve
            for j in range(velocity_samples):
                vel = lerp(
                    min_vel_solve[1],
                    max_shooter_velocity,
                    1 / (1.5 ** (velocity_samples - j - 1)),
                )
                # Feed previous solve into new solve as an initial guess
                solve = fixed_velocity(distance, vx, vy, vel, prev_solve[5])
                if solve[0]:
                    file.write(
                        f"        entry({vel}, new ShotResult("
                        f"{np.rad2deg(solve[2])},  {solve[3]}, {solve[4]}))"
                    )
                    prev_solve = solve
                    if j != velocity_samples - 1:
                        file.write(",\n")
                    else:
                        file.write("\n")
                else:
                    break
            file.write("      )\n")
            file.write("    );\n")

    file.write("  }\n\n")

    file.write("  @SafeVarargs\n")
    file.write("  private static InterpolatingTreeMap<Double, ShotResult> makeTable(\n")
    file.write("      Map.Entry<Double, ShotResult>... entries) {\n")
    file.write(
        "    var map = new InterpolatingTreeMap<>(MathUtil::inverseInterpolate, "
        "ShotResult::interpolate);\n"
    )
    file.write("    for (var entry : entries) {\n")
    file.write("      map.put(entry.getKey(), entry.getValue());\n")
    file.write("    }\n")
    file.write("    return map;\n")
    file.write("  }\n")

    file.write("}")
