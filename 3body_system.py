import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
from matplotlib.colors import cnames
import scipy.integrate

# Based on:
# https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767

#Define universal gravitation constant
G=6.67408e-11 #N-m2/kg2
#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri
#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#Define masses
m1=1.1 #Alpha Centauri A
m2=0.907 #Alpha Centauri B
m3=0.9 #Third Star

#It is like saying that stars have the same densities 
# so the volume of the marker inn the plot is proportional to its mass
sizes = [5*m1, 5*m2, 5*m3]

#Define initial position vectors
r1=[-0.5,0,0] #m
r2=[0.5,0,0] #m
r3=[0,1,0] #m

#Convert pos vectors to arrays
r1=sci.array(r1,dtype="float64")
r2=sci.array(r2,dtype="float64")
r3=sci.array(r3,dtype="float64")

#Find Centre of Mass
#r_com=(m1*r1+m2*r2)/(m1+m2)
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)

#Define initial velocities
v1=[0.01,0.01,0] #m/s
v2=[-0.05,0,-0.1] #m/s
v3=[0,-0.01,0]

#Convert velocity vectors to arrays
v1=sci.array(v1,dtype="float64")
v2=sci.array(v2,dtype="float64")
v3=sci.array(v3,dtype="float64")

#Find velocity of COM (center of mass)
#v_com=(m1*v1+m2*v2)/(m1+m2)
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)


def ThreeBodyEquations(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]

    r12=sci.linalg.norm(r2-r1)
    r13=sci.linalg.norm(r3-r1)
    r23=sci.linalg.norm(r3-r2)
    
    eq1=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    eq2=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    eq3=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    eq4=K2*v1
    eq5=K2*v2
    eq6=K2*v3

    r12_derivs=sci.concatenate((eq4,eq5))
    r_derivs=sci.concatenate((r12_derivs,eq6))
    v12_derivs=sci.concatenate((eq1,eq2))
    v_derivs=sci.concatenate((v12_derivs,eq3))

    derivs=sci.concatenate((r_derivs,v_derivs))

    return derivs

#Package initial parameters
init_params=sci.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array

time_span=sci.linspace(0,35,3500) #35 orbital periods and 3500 points 
#the ratio between the two set the velocity of the simulation

#Run the ODE solver
sol = sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

r1_t=sol[:,:3]
r2_t=sol[:,3:6]
r3_t=sol[:,6:9]

x1 = r1_t[:,0]
x2 = r2_t[:,0]
x3 = r3_t[:,0]
y1 = r1_t[:,1]
y2 = r2_t[:,1]
y3 = r3_t[:,1] 
z1 = r1_t[:,2]
z2 = r2_t[:,2]
z3 = r3_t[:,2]

#Packing them into a proper way for the animate function
data = np.array([[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]])

#Create figure
fig=plt.figure()
#Create 3D axes
ax=fig.add_subplot(111,projection="3d")

#Ax settings
ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_zlabel("z",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
#ax.legend(loc="upper left",fontsize=14)

#I don't know ho to rescale the axes in each frame so I set a custom one
xmin=x3.min(); xmax=x3.max()
ymin=y3.min(); ymax=y3.max()        
zmin=z3.min(); zmax=z3.max()
ax.set_xlim(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin))
ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
ax.set_zlim(zmin-0.1*(zmax-zmin),zmax+0.1*(zmax-zmin))

#ax.axis('off')
#fig.set_facecolor('black')
#ax.set_facecolor('black')

N_trajectories = 3
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

#Alternative delcaration of lines objects 
#lines = [ax.plot(dat[0, 0:2], dat[1, 0:2], dat[2, 0:2], ls ='-')[0] for dat in data]

lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])


#pts = sum([ax.plot([], [], [], 'o', c=c)
#           for c in colors], [])

#Creating points objects
varacaso = []
for c, s in zip (colors, sizes):
    varacaso.append(ax.plot([], [], [], 'o', c=c, markersize=s))

pts = sum(varacaso, [])     
    

# This sets inintial text (not working?)
#time_text = sum([ax.text(0.02, 0.95, 0.8, '')], [])

# This sets inintial cam view
#ax.view_init(30, 0)

def animate(i, data, lines, pts):

    for line, pt, data in zip(lines, pts, data):
        
        # nOTE: there is no .set_data() for 3 dim data...
        # set_3d_properties is the update of z axis
        if i > 150 :
            line.set_data(data[0:2, i-150:i])
            line.set_3d_properties(data[2, i-150:i])
        else :
            line.set_data(data[0:2, :i])
            line.set_3d_properties(data[2, :i])
        

        pt.set_data(data[0:2, i-2])
        pt.set_3d_properties(data[2, i-2])

        # camera moves
        #ax.view_init(30, 0.3 *i)
        #ax.autoscale_view()
        #fig.canvas.draw() #This makes the plot laggy but not when you save it
    
    #(Attempt to update a text object in each frame)
    #ax.text(0.02, 0.95, 0.8, 'time = %.1fy' % 0.51*i)
    #time_text.set_text('time = %.1fy' % 0.51*i)
           
    return  lines + pts  #, time_text (Attempt to update a text object in each frame)

"""
ANIMATION FUNCTION
   
   Update the data held by the scatter plot and therefore animates it.
   Args:
       iteration (int): Current iteration of the animation
       data (list): List of the data positions at each iteration.
       scatters (list): List of all the scatters (One per element)
   Returns:
       list: List of scatters (One per element) with new coordinates
"""

ani = animation.FuncAnimation(fig, animate, frames= 1000000,
                               blit=True, fargs = (data, lines, pts), repeat=False, interval = 0.1)#, init_func=init)

#To use the line below you need to install imagemagick
#ani.save('C:\\Users\\Salvo\\Desktop\\animation2.gif', writer='imagemagick', fps=60)
plt.show()


