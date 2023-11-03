import numpy as np
import math

def hfun(v0,rho):
    t0,s0,r0 = v0
    exprho = np.exp(rho)
    expnegrho = np.exp(-rho)
    f  = ((rho-1)*r0+s0)*exprho -     (r0-rho*s0)*expnegrho - (rho*(rho-1)+1)*t0
    df =     (rho*r0+s0)*exprho + (r0-(rho-1)*s0)*expnegrho -       (2*rho-1)*t0
    return f,df

def rootsearch_bn(fun, farg, xl, xh, x0): #dichotomy
    EPS = 10**(-10)
    assert xl<=x0<=xh
    xx = 0.5*(xl+xh)
    while abs(xx-x0) <= EPS:
        f = fun(farg,x0)[1]
        if f < 0.0 :
            xl = xx
        else:
            xh = xx
        xx = 0.5*(xl+xh)
    return xx


def rootsearch_ntinc(fun, farg, xl, xh, x0): #dichotomy
    EPS = 10**(-12)
    DFTOL = EPS**(6/7)
    MAXITER = 21
    LODAMP = 0.05
    HIDAMP = 0.95

    assert xl<=x0<=xh

    xx = x0
    converged = False

    for i in range(1,MAXITER):
        f,df = fun(farg,x0)
        if f < 0.0:
            xl = x0
        else:
            xh = x0

        if xh<=xl:
            converged = True
            break

        if math.isfinite(f) and df >= DFTOL:
            xx = x0 - f/df
        else:
            break

        if abs(xx-x0) <= EPS*max(1.,abs(xx)):
            converged = True
            break

        if xx>=xh:
            x0 = min(LODAMP*x0+HIDAMP*xh, xh)
        elif xx<=xl:
            x0 = max(LODAMP*x0+HIDAMP*xl, xl)
        else:
            x0 = xx

    if converged:
        return max(xl,min(xh,xx))
    else:
        return rootsearch_bn(fun,farg,xl,xh,x0)

def projheu_primalexpcone(v0):
    t0,s0,r0 = v0

    vp = [max(t0,0), 0.0, min(r0,0)]
    dist = np.linalg.norm(np.array(vp)-np.array(v0),2)

    if s0 > 0.0:
        tp = max(t0, s0*np.exp(r0/s0))
        newdist =  tp-t0
        if newdist < dist:
            vp = [tp, s0, r0]
            dist = newdist

    return [vp,dist]

def projheu_polarexpcone(v0):
    t0,s0,r0 = v0

    vd = [min(t0,0), min(s0,0), 0.0]
    dist = np.linalg.norm(np.array(vd)-np.array(v0),2)

    if r0 > 0.0:
        td = min(t0, -r0*np.exp((s0/r0)-1))
        newdist = t0-td
        if newdist < dist:
            vd = [td, s0, r0]
            dist  = newdist

    return [vd,dist]

def projsol_primalexpcone(v0, rho):
    t0,s0,r0 = v0

    linrho = ((rho-1)*r0+s0)

    exprho = np.exp(rho)
    if (linrho>0) and math.isfinite(exprho):
        quadrho=rho*(rho-1)+1
        vp=np.array([exprho,1,rho])*linrho/quadrho
        dist=np.linalg.norm(np.array(vp)-np.array(v0),2)
    else:
        vp = [np.inf,0.0,0.0]
        dist = np.inf

    return [vp,dist]

def projsol_polarexpcone(v0, rho):
    t0,s0,r0 = v0

    linrho = (r0-rho*s0)
    exprho = np.exp(-rho)
    if (linrho>0) and math.isfinite(exprho):
        quadrho=rho*(rho-1)+1
        vd=np.array([-exprho,1-rho,1])*linrho/quadrho
        dist=np.linalg.norm(np.array(vd)-np.array(v0),2)
    else:
        vd = [-np.inf,0.0,0.0]
        dist = np.inf

    return [vd,dist]

def ppsi(v0):
    t0,s0,r0 = v0

    if (r0 > s0):
        psi = (r0-s0 + np.sqrt(r0**2 + s0**2 - r0*s0)) / r0
    else:
        psi = -s0 / (r0-s0 - np.sqrt(r0**2 + s0**2 - r0*s0))

    return ((psi-1)*r0 + s0)/(psi*(psi-1) + 1)

def pomega(rho):
    val = np.exp(rho)/(rho*(rho-1)+1)
    if rho < 2.0:
        val = min(val, np.exp(2)/3)
    return val

def dpsi(v0):
    t0,s0,r0 = v0

    if( s0 > r0 ):
        psi = (r0 - np.sqrt(r0**2 + s0**2 - r0*s0)) / s0
    else:
        psi = (r0 - s0) / (r0 + np.sqrt(r0**2 + s0**2 - r0*s0))

    res = (r0 - psi*s0)/(psi*(psi-1) + 1)
    return res

def domega(rho):
    val = -np.exp(-rho)/(rho*(rho-1)+1)
    if rho > -1.0:
        val = max(val, -np.exp(1)/3)

    return val

def clamp(x,lo,hi):
  if lo <= x <= hi:return x
  if x<lo:return lo
  if x>hi:return hi


def searchbracket(v0, pdist, ddist):
    t0,s0,r0 = v0
    baselow,baseupr = -np.inf,np.inf
    low,upr = -np.inf,np.inf

    Dp = np.sqrt(pdist**2 - min(s0,0)**2)
    Dd = np.sqrt(ddist**2 - min(r0,0)**2)

    if (t0>0):
        tpl    = t0
        curbnd = np.log(tpl/ppsi(v0))
        low    = max(low,curbnd)

    if (t0<0):
        tdu    = t0
        curbnd = -np.log(-tdu/dpsi(v0))
        upr    = min(upr, curbnd)

    if (r0>0):
        baselow = 1-s0/r0
        low     = max(low, baselow)

        tpu    = max(1e-12, min(Dd, Dp+t0))
        palpha = low
        curbnd = max(palpha, baselow + tpu/r0/pomega(palpha))
        upr    = min(upr, curbnd)

    if (s0>0):
        baseupr = r0/s0
        upr     = min(upr, baseupr)

        tdl    = -max(1e-12, min(Dp, Dd-t0))
        dalpha = upr
        curbnd = min(dalpha, baseupr - tdl/s0/domega(dalpha))
        low    = max(low, curbnd)

    assert baselow <= baseupr
    assert math.isfinite(low)
    assert math.isfinite(upr)

    low,upr = min(low,upr),max(low,upr)
    low,upr = clamp(low,baselow,baseupr),clamp(upr,baselow,baseupr)
    return [low,upr]

def projprimalexpcone(v0):
    TOL = (10**(-10))**(2/3)
    t0,s0,r0 = v0

    vp,pdist = projheu_primalexpcone(v0)
    vd,ddist = projheu_polarexpcone(v0)
    if not  ( (s0<=0 and r0<=0) or min(pdist,ddist)<=TOL or ( np.linalg.norm(np.array(vp)+np.array(vd)-np.array(v0),np.inf)<=TOL and np.dot(np.array(vp),np.array(vd))<=TOL )):

          xl,xh = searchbracket(v0,pdist,ddist)
          rho   = rootsearch_ntinc(hfun,v0,xl,xh,0.5*(xl+xh))
          vp1,pdist1 = projsol_primalexpcone(v0,rho)
          if (pdist1 <= pdist):
            vp,pdist = vp1,pdist1

          vd1,ddist1 = projsol_polarexpcone(v0,rho)
          if (ddist1 <= ddist):
            vd,ddist = vd1,ddist1

    return [vp,vd]

def abserr(v0, vp, vd):
    return [np.linalg.norm(np.array(vp) + np.array(vd) - np.array(v0),2), abs(np.dot(vp,vd))]

def relerr(v0, vp, vd):
    return abserr(v0,vp,vd) / max(1.0,np.linalg.norm(np.array(v0),2))

def solutionreport(v0, vp, vd):
    print("abserr=",abserr(v0,vp,vd),"relerr=",relerr(v0,vp,vd),"v0=",v0,"vp=",vp,"vd=",vd)