"""Extended LQ thermoradiobiology (van Leeuwen et al. 2017; Kok et al. 2022/2024; Mingo Barba et al. 2024).
alpha(T,tint) = a37*exp[(T-37)/4 * ln(a41/a37) * exp(-mu*|tint|)], same for beta;
direct heat kill SF_HT = exp(-k(T)*t_HT), k = 2.05e10*(T+273.15)*exp(dS/2 - dH/(2(T+273.15))) [1/s].
EQD_RT = (alpha*D + G*beta*D^2)/(a37 + b37*dref) + k(T)*t_HT/(a37 + b37*dref)."""
import numpy as np
PAR = {  # maximum-likelihood estimates, van Leeuwen et al. 2017 (supplementary Tables B2/C2, centre values)
    "SiHa": dict(a37=0.386, ab37=17.914, ra=1.734, rb=0.418, mu=0.027, dS=392.082, dH=147907.8, sigma=0.273),
    "HeLa": dict(a37=0.230, ab37=4.999, ra=2.553, rb=0.577, mu=0.053, dS=432.361, dH=160105.6, sigma=0.292),
}
def lq(T, tint, p):
    T = np.asarray(T, float); f = (T - 37.0) / 4.0 * np.exp(-p["mu"] * np.abs(tint))
    a = p["a37"] * np.exp(f * np.log(p["ra"])); b = p["a37"] / p["ab37"] * np.exp(f * np.log(p["rb"]))
    return a, b
def k_heat(T, p):
    TK = np.asarray(T, float) + 273.15
    return 2.05e10 * TK * np.exp(p["dS"] / 2 - p["dH"] / (2 * TK))
def eqd(D, T, tint, p, dref=2.0, t_ht=3600.0, G=1.0, direct=True):
    a, b = lq(T, tint, p); a37 = p["a37"]; b37 = a37 / p["ab37"]
    e = (a * D + G * b * D**2) / (a37 + b37 * dref)
    if direct: e = e + k_heat(T, p) * t_ht / (a37 + b37 * dref)
    return e
if __name__ == "__main__":
    p = PAR["SiHa"]
    print("Appendix A check (SiHa, 4 Gy, 42C, tint 0, dref 2): no direct", round(float(eqd(4, 42, 0, p, direct=False)), 3), " with direct", round(float(eqd(4, 42, 0, p)), 3), " paper 7.43")
    for name, p in PAR.items():
        for T in (39, 40, 41, 41.2, 42, 43):
            print(name, T, "EQD of 2 Gy, tint 79 min:", round(float(eqd(2, T, 79/60, p)), 3), " direct part", round(float(k_heat(T, p) * 3600 / (p["a37"] + p["a37"]/p["ab37"]*2)), 3))
