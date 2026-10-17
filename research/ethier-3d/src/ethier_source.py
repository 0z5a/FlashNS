"""Ethier--Steinman transient 3D Beltrami solution, density=1.
Primary Eq.15 / independent Cardinal verification implementation. Pressure
p=-|u|^2/2; subtract a fixed spatial anchor at each time for training gauge.
No OpenAI construction or forcing is implied (force=0).
"""
import numpy as np

def velocity(points,a=.5,d=1.,nu=1.):
    x,y,z,t=np.moveaxis(np.asarray(points),-1,0);decay=np.exp(-nu*d*d*t)
    u=-a*(np.exp(a*x)*np.sin(a*y+d*z)+np.exp(a*z)*np.cos(a*x+d*y))*decay
    v=-a*(np.exp(a*y)*np.sin(a*z+d*x)+np.exp(a*x)*np.cos(a*y+d*z))*decay
    w=-a*(np.exp(a*z)*np.sin(a*x+d*y)+np.exp(a*y)*np.cos(a*z+d*x))*decay
    return np.stack((u,v,w),-1)

def fields(points,a=.5,d=1.,nu=1.,pressure_gauge=True):
    points=np.asarray(points);u=velocity(points,a,d,nu);p=-np.sum(u*u,-1)/2
    if pressure_gauge:
        anchor=np.zeros_like(points);anchor[...,3]=points[...,3];ref=velocity(anchor,a,d,nu);p-= -np.sum(ref*ref,-1)/2
    return np.concatenate((u,p[...,None]),-1)

def mp_fields(x,y,z,t,a,d,nu,pressure_gauge=True):
    import mpmath as mp
    e=mp.exp(-nu*d*d*t)
    u=-a*(mp.exp(a*x)*mp.sin(a*y+d*z)+mp.exp(a*z)*mp.cos(a*x+d*y))*e
    v=-a*(mp.exp(a*y)*mp.sin(a*z+d*x)+mp.exp(a*x)*mp.cos(a*y+d*z))*e
    w=-a*(mp.exp(a*z)*mp.sin(a*x+d*y)+mp.exp(a*y)*mp.cos(a*z+d*x))*e
    p=-(u*u+v*v+w*w)/2
    if pressure_gauge:p+=3*a*a*e*e/2
    return u,v,w,p
