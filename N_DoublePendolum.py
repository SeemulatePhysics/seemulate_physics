"""
N double pendulums 3D animation.
"""
# Generalization based on https://matplotlib.org/3.1.1/gallery/animation/double_pendulum_sgskip.html
# Equation used at the previous link can be found here http://www.physics.usyd.edu.au/~wheat/dpend_html/
# Equation also at but slightly different https://scipython.com/blog/the-double-pendulum/

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


#Equation of the double pendulum system
def derivs(theta, t):
  
    theta_dot = np.zeros_like(theta)
    theta_dot[0] = theta[1]

    delta = theta[2] - theta[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)

    theta_dot[1] = ((M2 * L1 * theta[1] * theta[1] * sin(delta) * cos(delta)
                + M2 * G * sin(theta[2]) * cos(delta)
                + M2 * L2 * theta[3] * theta[3] * sin(delta)
                - (M1+M2) * G * sin(theta[0]))
               / den1)

    theta_dot[2] = theta[3]

    den2 = (L2/L1) * den1

    theta_dot[3] = ((- M2 * L2 * theta[3] * theta[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(theta[0]) * cos(delta)
                - (M1+M2) * L1 * theta[1] * theta[1] * sin(delta)
                - (M1+M2) * G * sin(theta[2]))
               / den2)

    return theta_dot

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
duration = 20
N_frame = duration/dt
t = np.arange(0, duration, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 160.0001
w1 = 0.0
th2 = -10.0001
w2 = 0

N = 40 # N pendulums

#Arrays in which there will be inserted all the x, y positions for all the pendulums
# x[Nth pendulum][0 or 1 either you need the M1 or M2 x position][ith frame]
x = [  ]
y = [  ]

# Looping on N pendulums
for i in range (N):

    # initial state
    displacement = 0.001
    state = np.radians([th1, w1, th2 + displacement*i, w2])

    # integrate your ODE (ordinal differential equation) using scipy.integrate.
    sol = integrate.odeint(derivs, state, t)

    x1 = L1*sin(sol[:, 0])
    y1 = -L1*cos(sol[:, 0])

    x2 = L2*sin(sol[:, 2]) + x1
    y2 = -L2*cos(sol[:, 2]) + y1

    x.append([ x1 , x2 ])
    y.append([ y1 , y2 ])


# Creating figure
fig = plt.figure()

# Fig settings
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.set_title('Wait For it')
ax.grid()

# Creating lines objects to be updated (the pundulums figures)
varacaso = []
for i in range(N):
    varacaso.append(ax.plot([], [], 'o-'))

lines = varacaso   

# Creating text object to be updated
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

NN = range(N)

# Initial state of the pendulums
def init():
    for line in lines :
        line[0].set_data([], [])

    time_text.set_text('')

    return lines, time_text

"""
ANIMATION FUNCTION "animate(i)"
   Update the data held by the line objects and therefore animates it.
   Args:
       iteration (int): Current iteration of the animation
   Returns:
       list: new line object with new coordinates
"""

# x[Nth pendulum][0 or 1 either you need the M1 or M2 x position][ith frame]
def animate(i):

    # Looping on pendulums
    for n, line in zip (NN, lines):
        up_x = [0, x[n][0][i], x[n][1][i]]
        up_y = [0, y[n][0][i], y[n][1][i]]

        # Updating pendulums
        line[0].set_data(up_x, up_y)

    time_text.set_text(time_template % (i*dt))

    return lines, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(sol)),
                              interval= dt*1000, blit=False, init_func=init)

# Interval is the time between 2 consecutive frames in ms
# 3th argument is the number of frames

#For the saved animation the duration is going to be frames * (1 / fps) (in seconds) 
#For the display animation the duration is going to be frames * interval / 1000 (in seconds) 

#To use the line below you need to install imagemagick. There are other writers aviable
#ani.save('C:\\Users\\Salvo\\Desktop\\seemulate_physics.gif', writer='imagemagick', fps=50)

plt.show()
